using HDF5, GLMakie, Meshes, CoordRefSystems

function plothdf(h5file::String)
	h5 = h5open(h5file, "r")
	num_blocks::Int32 = read(HDF5.attributes(h5),"NumMeshBlocks")
	MeshBlockSize::Vector{Int32} = read(HDF5.attributes(h5),"MeshBlockSize")
	Br = read(h5["B"])[:,:,:,:,1]
	Bθ = read(h5["B"])[:,:,:,:,2]
	Bϕ::Array{Float32, 4} = read(h5["B"])[:,:,:,:,3]
	Bp = @. hypot(Br, Bθ)/(Bϕ)
	x_all::Array{Float32, 2} = read(h5["x1f"])   # shape: (Nx, Nblocks)
	y_all::Array{Float32, 2} = (read(h5["x2f"]))
	z_all::Array{Float32, 2} = (read(h5["x3f"]))
	close(h5)
	fig = Figure()
	ax::LScene = LScene(fig[1, 1], show_axis=false)

	function to_vertex_centered(Bt_block::Array{Float32,3})
		Nx, Ny, Nz = size(Bt_block)
		Bt_vertex = zeros(Float32, Nx+1, Ny+1, Nz+1)
		for i in 1:Nx, j in 1:Ny, k in 1:Nz
		    Bt_vertex[i, j, k] += Bt_block[i, j, k]
		    Bt_vertex[i+1, j, k] += Bt_block[i, j, k]
		    Bt_vertex[i, j+1, k] += Bt_block[i, j, k]
		    Bt_vertex[i, j, k+1] += Bt_block[i, j, k]
		    Bt_vertex[i+1, j+1, k] += Bt_block[i, j, k]
		    Bt_vertex[i+1, j, k+1] += Bt_block[i, j, k]
		    Bt_vertex[i, j+1, k+1] += Bt_block[i, j, k]
		    Bt_vertex[i+1, j+1, k+1] += Bt_block[i, j, k]
		end
		Bt_vertex ./= 8
	end
    #cr = log10.(extrema(abs.(Bp)))
    vlm = 0.0
    for b in 1:num_blocks
		print("\rBlock: $b ")
		try
			Bp_block = Bp[:, :, :, b]
			Bp_vertex = to_vertex_centered(Bp_block)
			nrho = abs.(vec(Bp_vertex))
			cre = extrema(nrho)
			r = x_all[:, b]
			θ = y_all[:, b]
			ϕ = z_all[:, b]
			g = RectilinearGrid{𝔼,typeof(Spherical(0,0,0))}((r, θ, ϕ))
			vlm += measure(g).val
			m = mapreduce(boundary, merge, g) |> Repair(0)
			viz!(m, color = nrho, alpha=0.1, colorrange = cre, colormap =:hawaii)
		catch e
		    if isa(e, ArgumentError)
		    	printstyled("ArgumentError in Block: $b \n", blink=true, color=:light_red)
		    else 
		    	printstyled("$e in Block: $b \n", blink=true, color=:light_red)
		    end
        end # trycatch
	end
	@show vlm
	fig
end
@time plothdf("sane00.prim.01800.athdf")
