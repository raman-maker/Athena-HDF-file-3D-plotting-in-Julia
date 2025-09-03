using HDF5, CairoMakie, Meshes, CoordRefSystems, LinearAlgebra, Base.Threads, Statistics

function plothdf(h5file::String)
    h5 = h5open(h5file, "r")
    num_blocks::Int32 = read(HDF5.attributes(h5), "NumMeshBlocks")
    MeshBlockSize::Vector{Int32} = read(HDF5.attributes(h5), "MeshBlockSize")

    Br = read(h5["B"])[:, :, :, :, 1]
    Bθ = read(h5["B"])[:, :, :, :, 2]
    Bϕ = read(h5["B"])[:, :, :, :, 3]
    @views Bp = hypot.(Br, Bθ) ./ Bϕ
    x_all = read(h5["x1f"])
    y_all = read(h5["x2f"])
    z_all = read(h5["x3f"])
    close(h5)

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = DataAspect())
	q_low  = quantile(vec(Bp), 0.05)
    q_high = quantile(vec(Bp), 0.95)
    cr = (q_low, q_high)

    for b in 1:num_blocks
        print("\rBlock: $b")

        @views Bp_block = Bp[18, :, :, b]
        r = @views x_all[:, b]
        θ = @views y_all[:, b]
        ϕ = @views z_all[:, b]

        g = RectilinearGrid{𝔼, typeof(Polar(0, 0))}(θ,ϕ)
        topo = topology(g)
        clr = Vector{Float64}(undef, nvertices(g))
        
        for i in 1:nvertices(g)
            adels = Coboundary{0,2}(topo)(i)	#Ntuple of element ids adjacent to this vertex
            clct  = collect(Bp_block[(elem2cart(topo,idx))...] for idx in adels)	#Position of i-th element in grid	
            clr[i] = mean(clct)
        end
        viz!(g, color = clr, colorrange = cr, colorscale = log10, colormap = :hawaii)
    end
    return fig
end

@time plothdf("sane98.prim.01800.athdf")
