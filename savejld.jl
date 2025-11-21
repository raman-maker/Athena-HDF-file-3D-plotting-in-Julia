using JLD2, GLMakie, Interpolations, Statistics, Meshes, CoordRefSystems, Unitful, Profile
using Base.Threads, Colors
using HDF5 # only for initial Athena++ read


function process_blocks_to_jld2(h5file; outfile = "final_volume_data.jld2", samples_per_axis::Int=6, blocks=1:872, grid_res=150)
    # Read large HDF5 dataset once
    h5 = h5open(h5file, "r")
    B = (h5["B"])
    Br, Bθ, Bϕ = @views B[:,:,:,:,1], B[:,:,:,:,2], B[:,:,:,:,3]
    Bp = @. hypot(Br, Bθ) / Bϕ
    levels = h5["Levels"]
    x_all, y_all, z_all = h5["x1f"], h5["x2f"], h5["x3f"]
    #close(h5)
    files = length(blocks)
	xmins, xmaxs, ymins, ymaxs, zmins, zmaxs = ntuple(_ -> fill(0f0, files*19*5*17), 6)
	
	icx=1
	for b in 1#blocks
        r, θ, ϕ = x_all[:, b], y_all[:, b], z_all[:, b]
        for rr in r, th in θ, ph in ϕ,  
			X = rr * sin(th) * cos(ph)
			Y = rr * sin(th) * sin(ph)
			Z = rr * cos(th)
			xmins[icx], xmaxs[icx] = extrema(X)
			ymins[icx], ymaxs[icx] = extrema(Y)
			zmins[icx], zmaxs[icx] = extrema(Z)
			icx += 1
			#@show extrema(X)
		end 
    end

	xmin=minimum(xmins); ymin=minimum(ymins); zmin=minimum(zmins)
    xmax=maximum(xmaxs); ymax=maximum(ymaxs); zmax=maximum(zmaxs)
    
   # println("🧭 Bounds:")
    println("  x ∈ [$xmin, $xmax], y ∈ [$ymin, $ymax], z ∈ [$zmin, $zmax]")
	# Pass 2: Allocate field
	nx = ny = nz = grid_res
	field = fill(NaN32, nx, ny, nz)
	println("🧩 Allocated $(nx)×$(ny)×$(nz) grid.")

	# Pass 3: Stream files lazily
	sx = (nx - 1) / (xmax - xmin)
	sy = (ny - 1) / (ymax - ymin)
	sz = (nz - 1) / (zmax - zmin)
	
    @threads for b in 1#blocks
    	println("Block: $b ")
        Bp_block = @views Bp[:, :, :, b]
        r, θ, ϕ = x_all[:, b], y_all[:, b], z_all[:, b]
        g = RectilinearGrid{𝔼, typeof(Spherical(0,0,0))}((r, θ, ϕ))
        
		# Compute total number of samples
		level::Int32 = (levels)[b]
		maxlvl = maximum(level)+1
		ns = (maxlvl - level) * samples_per_axis
		N_total = nelements(g)*ns^3
		Xlist, Ylist, Zlist, Clist = ntuple(_ -> Vector{Float32}(undef, N_total), 4)

		ibx::Int32 = 1
		iex::Int32 = 1
        for i in 1:nelements(g)
            el = element(g, i)
            clr = Vector{Float32}(undef, 8)
            bd = Boundary{3,0}(topology(g))(i)
            for (ii, id) in enumerate(bd)
                adels = Coboundary{0,3}(topology(g))(id)
                s = 0.0
                for iex in adels
                    inds = elem2cart(topology(g), iex)
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
				    Xlist[iex] = rr * sin(th) * cos(ph)
				    Ylist[iex] = rr * sin(th) * sin(ph)
				    Zlist[iex] = rr * cos(th)
				    Clist[iex] = val
				    iex += 1
				end
            end
        end   #elements end
		
		@inbounds for j in eachindex(Xlist)
			xi = Int32(round((Xlist[j] - xmin) * sx)) + 1
			yi = Int32(round((Ylist[j] - ymin) * sy)) + 1
			zi = Int32(round((Zlist[j] - zmin) * sz)) + 1
			ix = max(1, min(nx, xi))
			iy = max(1, min(ny, yi))
			iz = max(1, min(nz, zi))
			field[ix, iy, iz] = Clist[j]
		end
    # Save final volume
    
    end   #blocks end
    @save outfile field xmin xmax ymin ymax zmin zmax
end 

# ────────────────────────────────────────────────────────────────
# STEP 3: Reload precomputed JLD2 volume instantly
# ────────────────────────────────────────────────────────────────

function plot_saved_volume(outfile::String)
    @load outfile field xmin xmax ymin ymax zmin zmax
    finite_vals = filter(!isnan, vec(field))
    q_low, q_high = quantile(finite_vals, (0.05, 0.95))
    fig = Figure(size=(900, 900))
    #ax = Axis3(fig[1,1], title="Saved Volume", aspect=:data)
    ax = LScene(fig[1,1], show_axis=false)
    volume!(ax, xmin..xmax, ymin..ymax, zmin..zmax, field; colorrange=(q_low, q_high))
    display(fig)
    return fig
end

# ────────────────────────────────────────────────────────────────
# Example Usage
# ────────────────────────────────────────────────────────────────

# Step 1: Convert .athdf → .jld2 blocks
@profile process_blocks_to_jld2("sane00.prim.01800.athdf"; outfile = "final_volume_data.jld2", samples_per_axis=20, blocks=1:872, grid_res=50)

Profile.print(format=:flat)
# Step 3: Reload instantly
plot_saved_volume("final_volume_data.jld2")
