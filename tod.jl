using HDF5, CairoMakie, Meshes, CoordRefSystems, LinearAlgebra

function plothdf(h5file::String)
	h5 = h5open(h5file, "r")
	num_blocks::Int32 = read(HDF5.attributes(h5),"NumMeshBlocks")
	MeshBlockSize::Vector{Int32} = read(HDF5.attributes(h5),"MeshBlockSize")
	@show typeof(num_blocks)
	#@show levels = read(h5["Levels"])
	Br = read(h5["B"])[:,:,:,:,1]
	Bθ = read(h5["B"])[:,:,:,:,2]
	Bϕ::Array{Float32, 4} = read(h5["B"])[:,:,:,:,3]
	Bp = @. (Br+Bθ)/(Bϕ)
	#Bt::Array{Float32, 4} = @. sqrt(Br^2 + Bθ^2 + Bϕ^2)
	@show size(Bp)
	x_all::Array{Float32, 2} = read(h5["x1f"])   # shape: (Nx, Nblocks)
	y_all::Array{Float32, 2} = (read(h5["x2f"]))
	z_all::Array{Float32, 2} = (read(h5["x3f"]))
	close(h5)
	fig = Figure()
    ax::Axis = Axis(fig[1, 1], aspect = DataAspect())
    cr = extrema(log10.(abs.(Bp)))
    function to_vertex_centered_2D(data::Array{Float32,2})
		Nx, Ny = size(data)
		vertex_data = zeros(Float32, Nx+1, Ny+1)
		
		for i in 1:Nx, j in 1:Ny
		    vertex_data[i, j] += data[i, j]
		    vertex_data[i+1, j] += data[i, j]
		    vertex_data[i, j+1] += data[i, j]
		    vertex_data[i+1, j+1] += data[i, j]
		end
		vertex_data ./= 4
	end

    for b in 1:num_blocks
		print("\rBlock: $b ")
 		Bp_block = Bp[:, 2, :, b]
		Bp_vertex = to_vertex_centered_2D(Bp_block)
		nrho = abs.(vec(Bp_vertex))
		r = x_all[:, b]
		θ = y_all[:, b]
		ϕ = z_all[:, b]

		g = RectilinearGrid{𝔼,typeof(Polar(0,0))}(r, ϕ)
		viz!(g, color = nrho, colorrange = cr, colorscale=log10)
		#element(g,100)
	end
	fig
end
@timev plothdf("sane98.prim.01800.athdf")


