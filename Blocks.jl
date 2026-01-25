using HDF5, Interpolations, GLMakie, Base.Threads, Statistics

@time begin
function readhdf(h5file::String)
	h5 = h5open(h5file, "r")
	@views x1v::Matrix{Float32}, x2v::Matrix{Float32}, x3v::Matrix{Float32}= Array(h5["x1f"]), Array(h5["x2f"]), Array(h5["x3f"])
	prim = h5["prim"]
	@views ρ::Array{Float32, 4} = prim[:,:,:,:,1]
	B  = h5["B"]      # (Nx, Ny, Nz, blocks, 3)
	@views Br::Array{Float32, 4}, Bθ::Array{Float32, 4}, Bϕ::Array{Float32, 4} = B[:,:,:,:,1], B[:,:,:,:,2], B[:,:,:,:,3]
	Bp = @. hypot(Br, Bθ) / Bϕ
	blocks::Int32 = 400#read(HDF5.attributes(h5),"NumMeshBlocks") 
	Nblocks = 1:blocks
	Time::Float32 = read((HDF5.attributes(h5))["Time"])
	return Bp, ρ, x1v, x2v, x3v, blocks, Nblocks, Time
	close(h5)
end

#### Insert the location of data that you want to plot ####
Bp, ρ, x1v, x2v, x3v, blocks, Nblocks, Time = readhdf("/home/raman/Videos/DataSANE00LowRes/sane00.prim.01999.athdf")

function compute_bounds(x1v, x2v, x3v, Nblocks) 
    xmin=Inf32; xmax=-Inf32; ymin=Inf32; ymax=-Inf32; zmin=Inf32; zmax=-Inf32
    for b in Nblocks
        @views r, theta, phi = x1v[:, b], x2v[:, b], x3v[:, b]
        rvals = extrema(r)
        sinθs = extrema(sin, theta)
        cosθs = extrema(cos, theta)
        sinϕs = extrema(sin, phi)
        cosϕs = extrema(cos, phi)        
		new_xmin::Float32, new_xmax::Float32 = extrema(r*sinθ*cosϕ for r in rvals for sinθ in sinθs for cosϕ in cosϕs)
        xmax = max(xmax, new_xmax); 	xmin = min(xmin, new_xmin)
		new_ymin::Float32, new_ymax::Float32 = extrema(r*sinθ*sinϕ for r in rvals for sinθ in sinθs for sinϕ in sinϕs)
        ymax = max(ymax, new_ymax);		ymin = min(ymin, new_ymin)
        new_zmin::Float32, new_zmax::Float32 = extrema(r*cosθ for r in rvals for cosθ in cosθs)
        zmax = max(zmax, new_zmax);		zmin = min(zmin, new_zmin)
    end
    @show xmin, xmax, ymin, ymax, zmin, zmax
    return xmin, xmax, ymin, ymax, zmin, zmax
end

#### Specify the minimum amd maximum values of x,y,z coordinates values that you want to plot upto.
xmin, xmax, ymin, ymax, zmin, zmax = compute_bounds(x1v, x2v, x3v, Nblocks) # -30,30,-30,30,-30,30 #

function cartesian_to_spherical(x, y, z)
    r = hypot(x, y, z)
    θ = r == 0 ? 0.0 : acos(z/r)
	ϕ = mod2pi(atan(y, x))  # In [0, 2pi) (like ϕs)
    return r, θ, ϕ
end

function append_repeat_last(A::AbstractArray)
    B = similar(A, size(A) .+ 1)
    B[axes(A)...] .= A
    for I in CartesianIndices(A)
        B[I] = A[I]
    end
    for I in CartesianIndices(B)
        idx = ntuple(d -> min(I[d], size(A, d)), ndims(A))
        B[I] = A[idx...]
    end
    return B
end

function block_interpolator(x1,x2,x3, Bblock::Array{Float32, 3})
    extrapolate(interpolate((x1, x2, x3), Bblock, Gridded(Linear())), NaN32)
    #extrapolate(interpolate(Bblock, BSpline(Cubic())), NaN32) #,x1, x2, x3)
end

function interp(data, x1v, x2v, x3v, blocks, Nblocks)
	interps = Vector{Any}(undef, blocks)
	@threads for b in Nblocks
		@views begin
		    x1 = log.(x1v[:, b]); x2 = x2v[:, b]; x3 = x3v[:, b]
		  	data_block = data[:,:,:,b]
		    data_block = append_repeat_last(data_block)
			interps[b] = block_interpolator(x1,x2,x3, data_block)
		end
	end
	return interps
end

interps_Bp = interp(Bp, x1v, x2v, x3v, blocks, Nblocks)
interps_ρ = interp(ρ, x1v, x2v, x3v, blocks, Nblocks)
steps::Int32 = 15

function fill_field(xmin, xmax, ymin, ymax, zmin, zmax, Nblocks, steps, interps)
	@show rx= length(xmin:steps:xmax); ry= length(ymin:steps:ymax); rz = length(zmin:steps:zmax)
	field = fill(NaN32, (rx,ry,rz))
	#=for (i,x) in enumerate(xmin:steps:xmax), (j,y) in enumerate(ymin:steps:ymax), (k,z) in enumerate(zmin:steps:zmax)
		r, θ, ϕ = cartesian_to_spherical(x, y, z)
		for b in Nblocks
			itp = interps[b](log(r), θ, ϕ)
			itp !==NaN32 ? field[i,j,k]= itp : continue
		end
	end=#
	@threads for i in 1:rx
		x = xmin + (i-1)*steps
		for j in 1:ry
		    y = ymin + (j-1)*steps
		    for k in 1:rz
		        z = zmin + (k-1)*steps
		        r, θ, ϕ = cartesian_to_spherical(x, y, z)
		        lr = log(r)
		        @inbounds for b in Nblocks
		            v = interps[b](lr, θ, ϕ)
		            if !isnan(v)
		                field[i,j,k] = v
		                break
		            end
		        end
		    end
		end
	end

	return field
end

field_Bp = fill_field(xmin, xmax, ymin, ymax, zmin, zmax, Nblocks, steps, interps_Bp)
field_ρ = fill_field(xmin, xmax, ymin, ymax, zmin, zmax, Nblocks, steps, interps_ρ)

function tdplot(xmin, xmax, ymin, ymax, zmin, zmax, Time, field_Bp, field_ρ)
	finite_vals = filter(!isnan, vec(field_ρ))
	q_low, q_high = quantile(finite_vals, (0.05, 0.95))
	fig = Figure()
	ax::LScene = LScene(fig[1,1], show_axis=true)

	volume!(ax, xmin..xmax, ymin..ymax, zmin..zmax, field_Bp; transparency=true, algorithm = :iso, isorange =0.4, isovalue =2.6063323f0, colormap   =[:red])#mean(finite_vals))
	volume!(ax, xmin..xmax, ymin..ymax, zmin..zmax, field_ρ; transparency=true, colorscale=Makie.pseudolog10)#, colorrange=(q_low, q_high))
	#volume!(ax, xmin..xmax, ymin..ymax, zmin..zmax, field; algorithm = :absorption, absorption=2f0, colorscale=Makie.pseudolog10)
	#volume!(ax, xmin..xmax, ymin..ymax, zmin..zmax, field; transparency=true, algorithm = :additive, diffuse=0.05, colorrange=(q_low, q_high))
	update_cam!(ax.scene, cameracontrols(ax), Point3f(1500,1500,1000), Point3f(0, 0, 500), Vec3f(0, 0, 1))
	display(fig, update=false)
	save("zoom$Time.png", fig, update=false)
end

tdplot(xmin, xmax, ymin, ymax, zmin, zmax, Time, field_Bp, field_ρ)

end

