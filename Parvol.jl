using GLMakie, Random, Interpolations
using HDF5, Meshes, CoordRefSystems, Unitful, Statistics
using Base.Threads

function plothdf_volume_parallel(h5file::String; samples_per_axis::Int = 6, grid_res::Int = 100)
    # --- Load HDF5 data ---
    h5 = h5open(h5file, "r")
    num_blocks::Int32 = read(HDF5.attributes(h5), "NumMeshBlocks")
    MeshBlockSize::Vector{Int32} = read(HDF5.attributes(h5), "MeshBlockSize")
    B = read(h5["B"])
    Br, Bθ, Bϕ = @views B[:,:,:,:,1], B[:,:,:,:,2], B[:,:,:,:,3]
    Bp = @. hypot(Br, Bθ) / Bϕ
    x_all::Array{Float32, 2} = read(h5["x1f"])
    y_all::Array{Float32, 2} = read(h5["x2f"])
    z_all::Array{Float32, 2} = read(h5["x3f"])
    close(h5)

    # --- Visualization setup ---
    fig = Figure(size = (1000, 800))
    ax = Axis3(fig[1, 1], title = "Interpolated Volume", aspect = :data, viewmode = :free)

    println("Processing $num_blocks blocks with $(nthreads()) threads...")

    # --- Thread-local accumulators (stored in a Dict) ---
    accumulators = Dict{Int, NamedTuple{(:X, :Y, :Z, :C), NTuple{4, Vector{Float32}}}}()

    # --- Parallel over mesh blocks ---
    @threads for b in 2000#:num_blocks
        tid = threadid()
        acc = get!(accumulators, tid) do
            (X = Float32[], Y = Float32[], Z = Float32[], C = Float32[])
        end

        @views Bp_block = Bp[:, :, :, b]
        r = x_all[:, b]
        θ = y_all[:, b]
        ϕ = z_all[:, b]
        g = RectilinearGrid{𝔼, typeof(Spherical(0,0,0))}((r, θ, ϕ))

        for i in 1:nelements(g)
            el = element(g, i)

            # ---- Corner values ----
            clr = Vector{Float32}(undef, 8)
            bd = Boundary{3,0}(topology(g))(i)
            for (ii, id) in enumerate(bd)
                adels = Coboundary{0,3}(topology(g))(id)
                s = 0.0f0
                for idx in adels
                    inds = elem2cart(topology(g), idx)
                    s += Bp_block[inds...]
                end
                clr[ii] = s / length(adels)
            end

            # ---- Vertex coordinates ----
            verts = [Float64.(ustrip.((c.r, c.θ, c.ϕ))) for c in coords.(vertices(el))]
            rgrid = LinRange(extrema(getindex.(verts, 1))..., 2)
            θgrid = LinRange(extrema(getindex.(verts, 2))..., 2)
            ϕgrid = LinRange(extrema(getindex.(verts, 3))..., 2)

            # ---- Build 2×2×2 data cube ----
            data = Array{Float32}(undef, 2, 2, 2)
            data[1,1,1] = clr[1]; data[2,1,1] = clr[2]
            data[2,2,1] = clr[3]; data[1,2,1] = clr[4]
            data[1,1,2] = clr[5]; data[2,1,2] = clr[6]
            data[2,2,2] = clr[7]; data[1,2,2] = clr[8]

            itp = interpolate((rgrid, θgrid, ϕgrid), data, Gridded(Linear()))

            r_eval = LinRange(rgrid[1], rgrid[end], samples_per_axis)
            θ_eval = LinRange(θgrid[1], θgrid[end], samples_per_axis)
            ϕ_eval = LinRange(ϕgrid[1], ϕgrid[end], samples_per_axis)

            for rr in r_eval, th in θ_eval, ph in ϕ_eval
                val = itp(rr, th, ph)
                x = rr * sin(th) * cos(ph)
                y = rr * sin(th) * sin(ph)
                z = rr * cos(th)
                push!(acc.X, x)
                push!(acc.Y, y)
                push!(acc.Z, z)
                push!(acc.C, val)
            end
        end
    end

    println("\nInterpolation complete. Merging results...")

    # --- Merge all thread accumulators ---
    Xacc_all = reduce(vcat, [v.X for v in values(accumulators)])
    Yacc_all = reduce(vcat, [v.Y for v in values(accumulators)])
    Zacc_all = reduce(vcat, [v.Z for v in values(accumulators)])
    Cacc_all = reduce(vcat, [v.C for v in values(accumulators)])

    # --- Compute bounding box ---
    x_min, x_max = extrema(Xacc_all)
    y_min, y_max = extrema(Yacc_all)
    z_min, z_max = extrema(Zacc_all)

    nx = ny = nz = grid_res
    x_range = LinRange(x_min, x_max, nx)
    y_range = LinRange(y_min, y_max, ny)
    z_range = LinRange(z_min, z_max, nz)

    # --- Fill 3D grid with nearest-point assignment ---
    field = fill(NaN32, nx, ny, nz)
    for (xi, yi, zi, ci) in zip(Xacc_all, Yacc_all, Zacc_all, Cacc_all)
        ix = clamp(Int(round((xi - x_min)/(x_max - x_min) * (nx - 1))) + 1, 1, nx)
        iy = clamp(Int(round((yi - y_min)/(y_max - y_min) * (ny - 1))) + 1, 1, ny)
        iz = clamp(Int(round((zi - z_min)/(z_max - z_min) * (nz - 1))) + 1, 1, nz)
        field[ix, iy, iz] = ci
    end

    # --- Compute color normalization ---
    finite_vals = filter(!isnan, vec(field))
    q_low  = quantile(finite_vals, 0.05)
    q_high = quantile(finite_vals, 0.95)
    cr = (q_low, q_high)
    field[field .< q_low] .= NaN

    println("Rendering volume...")
    volume!(ax, x_min..x_max, y_min..y_max, z_min..z_max, field; colorrange=cr)
    fig
end

# --- Example usage ---
Threads.nthreads() == 1 && println("⚠️ Warning: Only 1 thread available. Run Julia with:  JULIA_NUM_THREADS=8 julia")
@time fig = plothdf_volume_parallel("sane98.prim.01900.athdf"; samples_per_axis = 20, grid_res = 150)
display(fig)
