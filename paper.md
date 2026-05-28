---
title: 'Athena.jl: A Julia package for 3D visualization of ATHENA++ AMR simulation data'
tags:
  - Julia
  - astrophysics
  - scientific visualization
  - ATHENA++
  - adaptive mesh refinement
  - magnetohydrodynamics
authors:
  - name: [Raman Kumar]
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: [SGMK, Poland]
    index: 1
date: 2026-03-13
bibliography: paper.bib
---

# Summary

`Athena.jl` is a Julia package for reading, processing, and rendering interactive 3D volumetric visualizations of output data produced by the ATHENA++ astrophysical magnetohydrodynamics (MHD) code [@Stone2020]. ATHENA++ employs adaptive mesh refinement (AMR), organizing simulation data into discrete "mesh blocks" stored in hierarchical HDF5 files. This structure presents a significant challenge for direct visualization: each mesh block covers a different region of space in the used coordinate system, and the blocks must be assembled, interpolated onto a Cartesian grid, and transformed before any volumetric rendering is possible. `Athena.jl` automates this entire pipeline — from HDF5 I/O to fully rendered 3D volume plots — leveraging Julia's performance and the GLMakie graphics library [@DanischKrumbiegel2021].

The package is aimed at researchers studying accretion flows, black hole magnetospheres, stellar interiors, and other astrophysical phenomena commonly simulated with ATHENA++. It produces publication-quality 3D renderings of both scalar fields (such as gas density ρ) and vector fields (such as the magnetic field B), with support for simultaneous multi-field overlays, isosurface extraction, and pseudolog color scaling etc.

# Statement of Need

ATHENA++ is one of the most widely used astrophysical MHD codes, and its simulation outputs are frequently used to study phenomena across many orders of magnitude in spatial scale — from the innermost regions of accretion disks to large-scale jets [@Porth2019]. Despite its widespread adoption, the visualization of AMR-structured ATHENA++ output remains non-trivial: the HDF5 format stores data block-by-block in Z-order curve, and no mature, open-source Julia-native tool existed for rendering this data in three dimensions.

Existing solutions typically require ParaView or VisIT software but sometimes they also gives error. It is very difficult to debug error in the ParaView or VisIT softwares due to their large size. But Julia code `Athena.jl` is easy to understand and customize. `Athena.jl` fills this gap by providing a self-contained, multithreaded pipeline that reads AMR block data directly, constructs block-level interpolators, maps the data onto a uniform Cartesian grid, and renders 3D volume and isosurface plots — all within a single Julia session. Researchers working primarily in Julia can easily make this code read Athena++ output data in other formats like History File(hst),Formatted Table, Restart(rst) and VTK etc.

# State of the Field

Volumetric visualization of astrophysical simulation data is a well-established need. Python packages such as yt [@Turk2011] provide comprehensive analysis tools for AMR codes and support ATHENA++, but require data to be loaded into Python's memory model. ParaView and VisIt offer powerful 3D visualization but can sometimes give error, have steep learning curves and require format-specific readers. Within the Julia ecosystem, Makie [@DanischKrumbiegel2021] provides modern, GPU-accelerated 3D rendering; however, no package previously bridged Makie's rendering capabilities with the ATHENA++ AMR block data format. `Athena.jl` is the first Julia package to do so, providing a lightweight and composable solution that integrates naturally with Julia's scientific stack.

# Software Design

`Athena.jl` is organized around four sequential stages:

**1. HDF5 I/O (`readhdf`)**: The package reads ATHENA++ `.athdf` files using HDF5.jl. It extracts the face-coordinate arrays (`x1f`, `x2f`, `x3f`) for each mesh block, the primitive variable array (density ρ), the magnetic field components (B_r, B_θ, B_ϕ), and global metadata such as the simulation time and total number of mesh blocks.

**2. Block-level interpolation (`interp`)**: Each mesh block covers a sub-domain in spherical coordinates (r, θ, ϕ). The package constructs a trilinear grid interpolator for each block using Interpolations.jl, operating in log-radial space (`log(r)`) to handle the large dynamic range typical of accretion disk simulations. A boundary correction step (`append_repeat_last`) ensures that cell-face coordinates are compatible with cell-centered data. All block interpolators are constructed in parallel using Julia's `Base.Threads` multi-threading.

**3. Cartesian grid assembly (`fill_field`)**: A uniform Cartesian grid is defined over a user-specified bounding box `[xmin, xmax] × [ymin, ymax] × [zmin, zmax]`. Note that larger step, lowers resolution.Each Cartesian grid point is converted to spherical coordinates (`cartesian_to_spherical`), and the appropriate block interpolator is queried. The assembly loop is multi-threaded and uses early-exit logic: once a non-NaN interpolation result is obtained from any block, the search terminates for that grid point.

**4. 3D rendering (`plot3d`)**: The assembled Cartesian volume arrays are rendered using GLMakie's `volume!` function. The density field is rendered with pseudolog color scaling to span the large dynamic range. The poloidal magnetic field is rendered simultaneously as an isosurface at its spatial mean value, providing an intuitive representation of the large-scale magnetic geometry. The resulting figure is displayed interactively and saved to a SVG or PNG file.

# Research Impact Statement

`Athena.jl` enables researchers to rapidly generate 3D visual diagnostics of ATHENA++ simulations, accelerating the scientific iteration cycle. Visual inspection of three-dimensional field structure — particularly for quantities like magnetic flux and density stratification — provides intuition that is difficult or impossible to obtain from 2D slices alone. By integrating directly with the Julia ecosystem, `Athena.jl` allows visualization to be embedded within broader analysis pipelines written in Julia, without requiring format conversion or interoperability layers. The package has been developed and tested on SANE (Standard and Normal Evolution) black hole accretion disk simulations [@Narayan2012], and is ready to use for any ATHENA++ simulation that uses spherical-polar coordinates with AMR.

# Acknowledgements

The author thanks the ATHENA++ development team for their open-source codebase and documentation. The Julia community is thanked for the HDF5.jl, Interpolations.jl, and GLMakie packages that this work depends upon.

# References

