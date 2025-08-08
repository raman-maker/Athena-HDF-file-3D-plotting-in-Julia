using HDF5, GLMakie, Colors

GLMakie.activate!(ssao = true)
GLMakie.closeall()

function plothdf_scatter(h5file::String)
    h5 = h5open(h5file, "r")
    num_blocks = read(HDF5.attributes(h5), "NumMeshBlocks")
    ρ = read(h5["prim"])[:,:,:,:,1]
    x_all = read(h5["x1v"])
    y_all = read(h5["x2v"])
    z_all = read(h5["x3v"])
    close(h5)

    fig = Figure(size = (1000, 800))
    ax = LScene(fig[1, 1]; show_axis = false)

    for b in 1:num_blocks
        r = x_all[:, b]
        θ = y_all[:, b]
        ϕ = z_all[:, b]
        Nx, Ny, Nz = length(r), length(θ), length(ϕ)
        X, Y, Z, V = Float32[], Float32[], Float32[], Float32[]
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            x = r[i] * sin(θ[j]) * cos(ϕ[k])
            y = r[i] * sin(θ[j]) * sin(ϕ[k])
            z = r[i] * cos(θ[j])
            push!(X, x)
            push!(Y, y)
            push!(Z, z)
            val = ρ[i, j, k, b]
            push!(V, (abs(val) + eps()))
        end
        cr = extrema(V)
        V ./= maximum(abs, V)  # normalize per block
        scatter!(ax, X, Y, Z; markersize = 1.5, color = V, colormap = :viridis, colorrange = cr, colorscale = log10)
    end
    fig
end
@timev plothdf_scatter("sane98.prim.01800.athdf")

