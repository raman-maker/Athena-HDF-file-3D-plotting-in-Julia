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
        
		# Compute total number of samples
		level::Int32 = read(h5["Levels"])[b]
		maxlvl = maximum(level)+1
		ns = (maxlvl - level) * samples_per_axis
		N_total = nelements(g)*ns^3
		Xlist, Ylist, Zlist, Clist = ntuple(_ -> Vector{Float32}(undef, N_total), 4)


		idx::Int32 = 1
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
            for rr in LinRange(rgrid[1], rgrid[end], ns),
                th in LinRange(θgrid[1], θgrid[end], ns),
                ph in LinRange(ϕgrid[1], ϕgrid[end], ns)
                val = itp(rr, th, ph)
               	@inbounds begin
				    Xlist[idx] = rr * sin(th) * cos(ph)
				    Ylist[idx] = rr * sin(th) * sin(ph)
				    Zlist[idx] = rr * cos(th)
				    Clist[idx] = val
				    idx += 1
				end
            end
        end

        x_min, x_max = extrema(Xlist)
		y_min, y_max = extrema(Ylist)
		z_min, z_max = extrema(Zlist)
		@save outfile Xlist Ylist Zlist Clist x_min x_max y_min y_max z_min z_max

        #println("💾 Saved block $b → $outfile")# ($(length(Xlist)) pts)")
    end
end


# ────────────────────────────────────────────────────────────────
# STEP 2: Stream JLD2 blocks into a unified 3D field (lazy loading)
# ────────────────────────────────────────────────────────────────

function stream_jld2_blocks(outdir; grid_res=150, outfile="final_volume.jld2")
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(outdir; join=true)))
    #isempty(files) && error("No .jld2 block files in $outdir")
    println("📦 Found $(length(files)) cached blocks")

    # Pass 1: Compute global bounds (streamed)
    println("🔍 Computing bounds ...")
	Nf = length(files)
	xmins, xmaxs, ymins, ymaxs, zmins, zmaxs = ntuple(_ -> Vector{Float32}(undef, Nf), 6)

	@inbounds for i in 1:Nf
		f = files[i]
		jldopen(f, "r") do jld
			xmins[i] = jld["x_min"]; xmaxs[i] = jld["x_max"]
			ymins[i] = jld["y_min"]; ymaxs[i] = jld["y_max"]
			zmins[i] = jld["z_min"]; zmaxs[i] = jld["z_max"]
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
        jldopen(f, "r"; mmaparrays=false) do jld
            X, Y, Z, C = jld["Xlist"], jld["Ylist"], jld["Zlist"], jld["Clist"]
            sx = (nx - 1) / (x_max - x_min)
			sy = (ny - 1) / (y_max - y_min)
			sz = (nz - 1) / (z_max - z_min)
			@inbounds @fastmath for j in eachindex(X)
				xi = Int32(round((X[j] - x_min) * sx)) + 1
				yi = Int32(round((Y[j] - y_min) * sy)) + 1
				zi = Int32(round((Z[j] - z_min) * sz)) + 1
				ix = max(1, min(nx, xi))
				iy = max(1, min(ny, yi))
				iz = max(1, min(nz, zi))
				field[ix, iy, iz] = C[j]
			end
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
    #ax = Axis3(fig[1,1], title="Saved Volume", aspect=:data)
    ax = LScene(fig[1,1], show_axis=false)
    volume!(ax, x_min..x_max, y_min..y_max, z_min..z_max, field; colorrange=(q_low, q_high))
    display(fig)
    return fig
end


# ────────────────────────────────────────────────────────────────
# Example Usage
# ────────────────────────────────────────────────────────────────

# Step 1: Convert .athdf → .jld2 blocks
@time process_blocks_to_jld2("sane98.prim.01900.athdf"; outdir="981900", samples_per_axis=20, blocks=2000)

# Step 2: Stream into 3D field (lazy JLD2 reads)
@profile fig = stream_jld2_blocks("981900"; grid_res=50)
Profile.print(format=:flat)
# Step 3: Reload instantly
plot_saved_volume("final_volume.jld2")
