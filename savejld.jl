using JLD2, GLMakie, Interpolations, Statistics, Meshes, CoordRefSystems, Unitful, Profile
using Base.Threads, Colors
using HDF5 # only for initial Athena++ read

# ────────────────────────────────────────────────────────────────
# STEP 1: Convert .athdf → many small .jld2 blocks (streamable)
# ────────────────────────────────────────────────────────────────

function process_blocks_to_jld2(h5file; outdir="cache_blocks", samples_per_axis::Int=6, blocks=1:872)
    isdir(outdir) || mkpath(outdir)

    # Read large HDF5 dataset once
    h5 = h5open(h5file, "r")
    B = read(h5["B"])
    Br, Bθ, Bϕ = @views B[:,:,:,:,1], B[:,:,:,:,2], B[:,:,:,:,3]
    Bp = @. hypot(Br, Bθ) / Bϕ
    x_all, y_all, z_all = read(h5["x1f"]), read(h5["x2f"]), read(h5["x3f"])
    #close(h5)
		
    @threads for b in blocks
        outfile = joinpath(outdir, "block_$(lpad(b,4,'0')).jld2")
        if isfile(outfile)
            @info "Skipping existing $outfile"
            continue
        end

        Bp_block = @views Bp[:, :, :, b]
        r, θ, ϕ = x_all[:, b], y_all[:, b], z_all[:, b]
        g = RectilinearGrid{𝔼, typeof(Spherical(0,0,0))}((r, θ, ϕ))

        Xlist = Float32[]; Ylist = Float32[]; Zlist = Float32[]; Clist = Float32[]

        for i in 1:nelements(g)
            el = element(g, i)
            clr = Vector{Float32}(undef, 8)
            bd = Boundary{3,0}(topology(g))(i)
            for (ii, id) in enumerate(bd)
                adels = Coboundary{0,3}(topology(g))(id)
                s = 0.0
                for idx in adels
                    inds = elem2cart(topology(g), idx)
                    s += Bp_block[inds...]
                end
                clr[ii] = s / length(adels)
            end

            verts = [Float64.(ustrip.((c.r, c.θ, c.ϕ))) for c in coords.(vertices(el))]
            rgrid = LinRange(extrema(getindex.(verts, 1))..., 2)
            θgrid = LinRange(extrema(getindex.(verts, 2))..., 2)
            ϕgrid = LinRange(extrema(getindex.(verts, 3))..., 2)
            data = reshape(clr, 2, 2, 2)
            itp = interpolate((rgrid, θgrid, ϕgrid), data, Gridded(Linear()))
			level::Int32 = read(h5["Levels"])[b]
            for rr in LinRange(rgrid[1], rgrid[end], (3-level)*(samples_per_axis)),
                th in LinRange(θgrid[1], θgrid[end], (3-level)*(samples_per_axis)),
                ph in LinRange(ϕgrid[1], ϕgrid[end], (3-level)*(samples_per_axis))
                val = itp(rr, th, ph)
                push!(Xlist, rr * sin(th) * cos(ph))
                push!(Ylist, rr * sin(th) * sin(ph))
                push!(Zlist, rr * cos(th))
                push!(Clist, val)
            end
        end

        # Save to JLD2
        @save outfile Xlist Ylist Zlist Clist
        #println("💾 Saved block $b → $outfile")# ($(length(Xlist)) pts)")
    end
end


# ────────────────────────────────────────────────────────────────
# STEP 2: Stream JLD2 blocks into a unified 3D field (lazy loading)
# ────────────────────────────────────────────────────────────────

function stream_jld2_blocks(outdir; grid_res=150, outfile="final_volume.jld2")
    files = sort(filter(f -> endswith(f, "72.jld2"), readdir(outdir; join=true)))
    isempty(files) && error("No .jld2 block files in $outdir")
    println("📦 Found $(length(files)) cached blocks")

    # Pass 1: Compute global bounds (streamed)
    println("🔍 Computing bounds ...")
    xmins = Float32[]; xmaxs = Float32[]
    ymins = Float32[]; ymaxs = Float32[]
    zmins = Float32[]; zmaxs = Float32[]

    for f in files
        jldopen(f, "r"; mmaparrays=true) do jld
            X = jld["Xlist"]; Y = jld["Ylist"]; Z = jld["Zlist"]
            push!(xmins, minimum(X)); push!(xmaxs, maximum(X))
            push!(ymins, minimum(Y)); push!(ymaxs, maximum(Y))
            push!(zmins, minimum(Z)); push!(zmaxs, maximum(Z))
        end
    end
    x_min, x_max = minimum(xmins), maximum(xmaxs)
    y_min, y_max = minimum(ymins), maximum(ymaxs)
    z_min, z_max = minimum(zmins), maximum(zmaxs)
    println("🧭 Bounds:")
    println("  x ∈ [$x_min, $x_max], y ∈ [$y_min, $y_max], z ∈ [$z_min, $z_max]")

    # Pass 2: Allocate field
    nx = ny = nz = grid_res
    field = fill(NaN32, nx, ny, nz)
    println("🧩 Allocated $(nx)×$(ny)×$(nz) grid.")

    # Pass 3: Stream files lazily
    for (i, f) in enumerate(files)
        println("📖 $(i)/$(length(files)): $(basename(f))")
        jldopen(f, "r"; mmaparrays=true) do jld
            X, Y, Z, C = jld["Xlist"], jld["Ylist"], jld["Zlist"], jld["Clist"]
            sx = (nx - 1) / (x_max - x_min)
			sy = (ny - 1) / (y_max - y_min)
			sz = (nz - 1) / (z_max - z_min)

			@inbounds @fastmath for j in eachindex(X)
				xi = Int(round((X[j] - x_min) * sx)) + 1
				yi = Int(round((Y[j] - y_min) * sy)) + 1
				zi = Int(round((Z[j] - z_min) * sz)) + 1
				ix = max(1, min(nx, xi))
				iy = max(1, min(ny, yi))
				iz = max(1, min(nz, zi))
				field[ix, iy, iz] = C[j]
			end

            #=@inbounds for j in eachindex(X)
                xi, yi, zi, ci = X[j], Y[j], Z[j], C[j]
                ix = (Int(round((xi - x_min)/(x_max - x_min) * (nx - 1))) + 1, 1, nx)   #clamp
                iy = (Int(round((yi - y_min)/(y_max - y_min) * (ny - 1))) + 1, 1, ny)
                iz = (Int(round((zi - z_min)/(z_max - z_min) * (nz - 1))) + 1, 1, nz)
                field[ix, iy, iz] = ci
            end=#
        end
    end
    println("✅ Streaming complete.")

    # Save final volume
    @save outfile field x_min x_max y_min y_max z_min z_max
    println("💾 Saved → $outfile")

    # Plot
    finite_vals = filter(!isnan, vec(field))
    q_low, q_high = quantile(finite_vals, (0.05, 0.95))
    fig = Figure(size=(900, 900))
    ax = LScene(fig[1,1], show_axis=false)
    volume!(ax, x_min..x_max, y_min..y_max, z_min..z_max, field; colorrange=(q_low, q_high))
    display(fig)
    return fig
end


# ────────────────────────────────────────────────────────────────
# STEP 3: Reload precomputed JLD2 volume instantly
# ────────────────────────────────────────────────────────────────

function plot_saved_volume(outfile::String)
    @load outfile field x_min x_max y_min y_max z_min z_max
    finite_vals = filter(!isnan, vec(field))
    q_low, q_high = quantile(finite_vals, (0.05, 0.95))
    fig = Figure(size=(900, 900))
    ax = Axis3(fig[1,1], title="Saved Volume", aspect=:data)
    volume!(ax, x_min..x_max, y_min..y_max, z_min..z_max, field; colorrange=(q_low, q_high))
    display(fig)
    return fig
end


# ────────────────────────────────────────────────────────────────
# Example Usage
# ────────────────────────────────────────────────────────────────

# Step 1: Convert .athdf → .jld2 blocks
@time process_blocks_to_jld2("sane00.prim.01800.athdf"; samples_per_axis=20, blocks=1:872)

# Step 2: Stream into 3D field (lazy JLD2 reads)
fig = stream_jld2_blocks("cache_blocks"; grid_res=150)
#Profile.print(format=:flat)
# Step 3: Reload instantly
plot_saved_volume("final_volume.jld2")

