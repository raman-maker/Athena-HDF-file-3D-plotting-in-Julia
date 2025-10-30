using GLMakie, Random, Interpolations
using HDF5, GLMakie, Meshes, CoordRefSystems, Unitful, Statistics

# Interpolate color values inside the hexahedron manually via trilinear interpolation
# Create normalized coordinates
function trilinear_interpolation(x, y, z, xgrid, ygrid, zgrid, v)
	# normalized coordinates (ξ, η, ζ) in [0,1]
	ξ = (x - xgrid[1]) / (xgrid[end] - xgrid[1])
	η = (y - ygrid[1]) / (ygrid[end] - ygrid[1])
	ζ = (z - zgrid[1]) / (zgrid[end] - zgrid[1])
	# vertex order (same as typical hexahedron corner order)
	v000, v100, v110, v010, v001, v101, v111, v011 = v
	# trilinear interpolation
	return (v000*(1-ξ)*(1-η)*(1-ζ) + v100*ξ*(1-η)*(1-ζ) + v110*ξ*η*(1-ζ) + v010*(1-ξ)*η*(1-ζ) + v001*(1-ξ)*(1-η)*ζ + v101*ξ*(1-η)*ζ 
			+ v111*ξ*η*ζ + v011*(1-ξ)*η*ζ)
end

function plothdf(h5file::String)
	h5 = h5open(h5file, "r")
	num_blocks::Int32 = read(HDF5.attributes(h5),"NumMeshBlocks")
	MeshBlockSize::Vector{Int32} = read(HDF5.attributes(h5),"MeshBlockSize")
	B = read(h5["B"])
	Br, Bθ, Bϕ = @views B[:,:,:,:,1], B[:,:,:,:,2], B[:,:,:,:,3]
	Bp = @. hypot(Br, Bθ)/(Bϕ)
	x_all::Array{Float32, 2} = read(h5["x1f"])   # shape: (Nx, Nblocks)
	y_all::Array{Float32, 2} = (read(h5["x2f"]))
	z_all::Array{Float32, 2} = (read(h5["x3f"]))
	close(h5)
	fig = Figure(size = (800, 800))
	ax = Axis3(fig[1, 1], title="Interpolated Color Volume in Spherical Hexahedron", aspect=:data, viewmode=:free)

	q_low  = quantile(vec(Bp), 0.15)  # 2% percentile
	q_high = quantile(vec(Bp), 0.65)  # 98% percentile
	cr = (q_low, q_high)

    for b in 2000#:num_blocks
		print("\rBlock: $b ")
		Bp_block = Bp[:, :, :, b]
		r = x_all[:, b]
		θ = y_all[:, b]
		ϕ = z_all[:, b]
		g = RectilinearGrid{𝔼,typeof(Spherical(0,0,0))}((r, θ, ϕ))
		
		for i in 1:nelements(g)
			el= element(g,i)

			clr = Float32[]
			bd = Boundary{3,0}(topology(g))(i)	#Ntuple of vertex ids of i-th element
			for id in bd
				adels= Coboundary{0,3}(topology(g))(id)	#Ntuple of element ids adjacent to this vertex
				clct = collect(Bp_block[(elem2cart(topology(g),idx))...] for idx in adels)	#position of i-th element in grid
				push!(clr, sum(clct)/length(clct))
			end

			verts = [Float64.(ustrip.((c.r, c.θ, c.ϕ))) for c in coords.(vertices(el))]

			# 2. Discretize grids in (r, θ, ϕ) space
			rgrid, θgrid, ϕgrid = (LinRange(extrema(getindex.(verts, i))..., 10) for i in 1:3)

			# 5. Compute interpolated data on regular (r,θ,ϕ) grid
			data = [trilinear_interpolation(r, θ, ϕ, rgrid, θgrid, ϕgrid, clr) for r in rgrid, θ in θgrid, ϕ in ϕgrid]

			# 6. Build interpolation object for continuous sampling
			itp = interpolate((rgrid, θgrid, ϕgrid), data, Gridded(Linear()))

			# 7. Define finer spherical grid for plotting
			r_eval = LinRange(rgrid[1], rgrid[end], 10)
			θ_eval = LinRange(θgrid[1], θgrid[end], 10)
			ϕ_eval = LinRange(ϕgrid[1], ϕgrid[end], 10)

			# Evaluate interpolated color data
			interp_data = Array{Float32}(undef, length(r_eval), length(θ_eval), length(ϕ_eval))
			for (i, r) in enumerate(r_eval), (j, θ) in enumerate(θ_eval), (k, ϕ) in enumerate(ϕ_eval)
				interp_data[i, j, k] = itp(r, θ, ϕ)
			end


			# 8. Convert (r, θ, ϕ) → Cartesian (x, y, z)
			xvals = [r * sin(θ) * cos(ϕ) for r in r_eval, θ in θ_eval, ϕ in ϕ_eval]
			yvals = [r * sin(θ) * sin(ϕ) for r in r_eval, θ in θ_eval, ϕ in ϕ_eval]
			zvals = [r * cos(θ)          for r in r_eval, θ in θ_eval, ϕ in ϕ_eval]
			
			# Scatter plot of true spherical geometry, colored by interpolated values
			scatter!(ax, vec(xvals), vec(yvals), vec(zvals),  color = vec(interp_data), markersize=8, colorrange=cr, transparency=true)
		end
	end
	fig
end
@time plothdf("sane98.prim.01900.athdf")

