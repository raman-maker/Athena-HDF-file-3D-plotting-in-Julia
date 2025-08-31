using HDF5, GLMakie, Meshes, CoordRefSystems, Statistics, Base.Threads, BenchmarkTools

function process_block(b, Bp, x_all, y_all, z_all, cr)
    try
        Bp_block = Bp[:, :, :, b]

        r = x_all[:, b]
        θ = y_all[:, b]
        ϕ = z_all[:, b]
        g = RectilinearGrid{𝔼,typeof(Spherical(0,0,0))}((r, θ, ϕ))
        vlm = measure(g).val
        m = mapreduce(boundary, merge, g) |> Repair(0)

        clr = Vector{Float64}(undef, nvertices(g))
        clrm = Vector{Float64}(undef, nvertices(m))
        gvs = vertices(g)
        mvs = vertices(m)

        # map mesh vertices
        m_dict = Dict{eltype(mvs), Vector{Int}}()
        for (j,v) in enumerate(mvs)
            push!(get!(m_dict, v, Int[]), j)
        end
        matches = [(i, j) for (i,v) in enumerate(gvs) if haskey(m_dict, v) for j in m_dict[v]]

        # compute vertex values (serial)
        for i in 1:nvertices(g)
            adels = Coboundary{0,3}(topology(g))(i)
            clct  = collect(Bp_block[(elem2cart(topology(g),idx))...] for idx in adels)
            clr[i] = mean(clct)
        end

        for i in 1:nvertices(g)
            clrm[i] = clr[get(matches, i, Int[])[2]]
        end

        return (m, clrm, vlm)
    catch e
        return (nothing, nothing, 0.0, e)
    end
end


function plothdf(h5file::String)
    h5 = h5open(h5file, "r")
    num_blocks::Int32 = read(HDF5.attributes(h5),"NumMeshBlocks")
    Br = read(h5["B"])[:,:,:,:,1]
    Bθ = read(h5["B"])[:,:,:,:,2]
    Bϕ::Array{Float32, 4} = read(h5["B"])[:,:,:,:,3]
    Bp = @. hypot(Br, Bθ)/(Bϕ)
    x_all::Array{Float32, 2} = read(h5["x1f"])
    y_all::Array{Float32, 2} = read(h5["x2f"])
    z_all::Array{Float32, 2} = read(h5["x3f"])
    close(h5)

    q_low  = quantile(vec(Bp), 0.20)
    q_high = quantile(vec(Bp), 0.80)
    cr = (q_low, q_high)
    @show cr

    # parallel computation of blocks
    results = Vector{Any}(undef, num_blocks)
    @threads for b in 1:num_blocks
        results[b] = process_block(b, Bp, x_all, y_all, z_all, cr)
    end
	
    # serial plotting
    vlm = 0.0
    fig = Figure()
    ax::LScene = LScene(fig[1, 1], show_axis=false)


	for (iblock, res) in enumerate(results)
		if length(res) == 4
		    println("Error in block number: $iblock")
		    continue
		end
		m, clrm, v = res
		vlm += v
		viz!(m, color = clrm, alpha=0.3, colorrange = cr, colormap = :hawaii)
	end


    @show vlm
    return fig
end

@time plothdf("sane00.prim.01800.athdf")

