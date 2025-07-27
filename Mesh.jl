using HDF5, GLMakie, Meshes, CoordRefSystems

function plothdf(h5file::String)
	h5 = h5open(h5file, "r")
	num_blocks::Int32 = read(HDF5.attributes(h5),"NumMeshBlocks")
	MeshBlockSize::Vector{Int32} = read(HDF5.attributes(h5),"MeshBlockSize")
	Bx = read(h5["B"])[:,:,:,:,1]
	By = read(h5["B"])[:,:,:,:,2]
	Bz::Array{Float32, 4} = read(h5["B"])[:,:,:,:,3]
	Bt = similar(Bx)
	Bt::Array{Float32, 4} = @. sqrt(Bx^2 + By^2 + Bz^2)
	x_all::Array{Float32, 2} = read(h5["x1f"])   # shape: (Nx, Nblocks)
	y_all::Array{Float32, 2} = (read(h5["x2f"]))
	z_all::Array{Float32, 2} = (read(h5["x3f"]))
	close(h5)
	fig = Figure()
	pl = PointLight(RGBf(20, 20, 20), Point3f(0))
	al = AmbientLight(RGBf(0.2, 0.2, 0.2))
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

    for b in num_blocks
		print("\rBlock: $b ")
		Bt_block = Bt[:, :, :, b]
		Bt_vertex = to_vertex_centered(Bt_block)
		nrho = log10.(abs.(vec(Bt_vertex)))
		r = x_all[:, b]
		θ = y_all[:, b]
		ϕ = z_all[:, b]
		g = RectilinearGrid{𝔼,typeof(Spherical(0,0,0))}((r, θ, ϕ))
		#m = mapreduce(boundary, merge, g) |> Repair(0)
		@show size(nrho)
		@show nelements(g)
		@show nvertices(g)
		viz!(g, color = nrho, alpha=0.9)
	end
	fig
end
@timev plothdf("sane00.prim.01800.athdf")

