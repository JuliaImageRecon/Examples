#---------------------------------------------------------
# # [NUFFT Overview](@id 1-nufft)
#---------------------------------------------------------

# This example illustrates how to use Nonuniform FFT (NUFFT)
# for image reconstruction in MRI
# using the Julia language.

# This entire page was generated using a single Julia file:
# [1-nufft.jl](https://github.com/JuliaImageRecon/Examples/blob/main/docs/lit/mri/1-nufft.jl).
# In any such Julia documentation,
# you can access the source code
# using the "Edit on GitHub" link in the top right.

# Some MRI scans use non-Cartesian sampling patterns
# (like radial and spiral k-space trajectories, among others),
# often in the interest of acquisition speed.

# Image reconstruction from fully sampled Cartesian k-space data
# typically uses simple inverse FFT operations,
# whereas non-Cartesian sampling requires more complicated methods.

# The examples here illustrate both non-iterative (aka "gridding")
# and iterative methods for non-Cartesian MRI reconstruction.
# For simplicity the examples consider the case of single-coil data
# and ignore the effects of B0 field inhomogeneity.


# First we add the Julia packages that are need for these examples.
# Change `false` to `true` in the following code block
# if you are using any of the following packages for the first time.

if false
    import Pkg
    Pkg.add([
        "ImagePhantoms"
        "Plots"
        "Unitful"
        "UnitfulRecipes"
        "LaTeXStrings"
        "MIRTjim"
        "MIRT"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ImagePhantoms: shepp_logan, SheppLoganToft, spectrum, phantom # Gauss2
using Plots; default(label="", markerstrokecolor=:auto)
using Unitful: mm
using UnitfulRecipes
using LaTeXStrings
using MIRTjim: jim, prompt
using MIRT: Anufft
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


# ## Radial sampling

# We focus on radial sampling as a simple representative non-Cartesian case.
# Consider imaging a 256mm × 256mm field of FOV
# with the goal of reconstructing a 128 × 128 pixel image.
# The following radial and angular k-space sampling is reasonable.

N = 128
FOV = 256mm # physical units!
Δx = FOV / N # pixel size
kmax = 1 / 2Δx
kr = ((-N÷2):(N÷2)) / (N÷2) * kmax # radial sampling in k-space
Nϕ = 3N÷2 # theoretically should be about π/2 * N
kϕ = (0:Nϕ-1)/Nϕ * π # angular samples
νx = kr * cos.(kϕ)' # N × Nϕ k-space sampling in cycles/mm
νy = kr * sin.(kϕ)'
plot(νx, νy,
    xlabel=L"\nu_x", ylabel=L"\nu_y",
    aspect_ratio = 1,
    title = "Radial k-space sampling",
)

#
isinteractive() && prompt();


# For the NUFFT routines considered here,
# the sampling arguments must be "Ω" values:
# digital frequencies have pseudo-units of radians / pixel.

Ωx = (2π * Δx) * νx # N × Nϕ grid of k-space sample locations
Ωy = (2π * Δx) * νy # in pseudo-units of radians / sample

scatter(Ωx, Ωy,
    xlabel=L"\Omega_x", ylabel=L"\Omega_y",
    xticks=((-1:1)*π, ["-π", "0", "π"]),
    yticks=((-1:1)*π, ["-π", "0", "π"]),
    xlims=(-π,π) .* 1.1,
    ylims=(-π,π) .* 1.1,
    aspect_ratio = 1, markersize = 0.5,
    title = "Radial k-space sampling",
)

#
isinteractive() && prompt()


# ## Radial k-space data for Shepp-Logan phantom

# Get the ellipse parameters for a MRI-suitable version of the Shepp-Logan phantom
# and calculate (analytically) the radial k-space data.
# Then display in polar coordinates.

object = shepp_logan(SheppLoganToft(); fovs=(FOV,FOV))
#object = [Gauss2(18mm, 0mm, 100mm, 70mm, 0, 1)] # useful for validating DCF
data = spectrum(object).(νx,νy)
data = data / oneunit(eltype(data)) # abandon units at this point
jim(kr, kϕ, abs.(data), title="k-space data magnitude",
    xlabel=L"k_r",
    ylabel=L"k_{\phi}",
    xticks = (-1:1) .* maximum(abs, kr),
    yticks = (0,π),
    ylims = (0,π),
    aspect_ratio = :none,
)


# ## Non-iterative gridding image reconstruction

# It would be impossible for a radiologist
# to diagnose a patient
# from the k-space data in polar coordinates,
# so image reconstruction is needed.

# The simplest approach
# is to (nearest-neighbor) interpolate the k-space data
# onto a Cartesian grid,
# and then reconstruction with an inverse FFT.

# One way to do that interpolation
# is to use a `Histogram` method
# in Julia's statistics package.

using StatsBase: fit, Histogram, weights

# The following function is a work-around because `weights` in StatsBase
# is
# [limited to Real data](https://github.com/JuliaStats/StatsBase.jl/issues/745),
# so here we bin the real and imaginary k-space data separately,
# and handle the units.
function histogram(coord, vals::AbstractArray{<:Number}, edges)
    u = oneunit(eltype(vals)) 
    wr = weights(real(vec(vals / u)))
    wi = weights(imag(vec(vals / u)))
    tmp1 = fit(Histogram, coord, wr, edges)
    tmp2 = fit(Histogram, coord, wi, edges)
    return u * complex.(tmp1.weights, tmp2.weights)
end

kx = N * Δx * νx # N × Nϕ k-space sampling in cycles/mm
ky = N * Δx * νy
bin = (-(N÷2):(N÷2)) .- 0.5 # (N+1,) histogram bin edges
gridded1 = histogram((vec(kx), vec(ky)), data, (bin,bin))

using FFTW: ifft, fftshift
tmp = fftshift(ifft(fftshift(gridded1)))
x = (-(N÷2):(N÷2-1)) * Δx
y = (-(N÷2):(N÷2-1)) * Δx
jim(x, y, tmp, title="Elementary gridding reconstruction")

# That crummy gridding method does not work well.
# Here's what the phantom should look like:

ideal = phantom(x, y, object, 2)
p0 = jim(x, y, ideal, title="True Shepp-Logan phantom")


# ## NUFFT approach to gridding

# Basic nearest-neighbor gridding
# does not provide acceptable image quality in MRI,
# so now we turn to using the NUFFT.
# For MRI purposes,
# the NUFFT is function that maps Cartesian spaced image data
# into non-Cartesian k-space data.
# The
# [NFFT.jl](https://github.com/tknopp/NFFT.jl)
# package has functions
# for computing the NUFFT and its adjoint.
# These are linear mappings (generalizations of matrices),
# so instead of calling those functions directly,
# here we use the NUFFT linear map object
# defined in MIRT
# that provides a non-Cartesian Fourier encoding "matrix".

A = Anufft([vec(Ωx) vec(Ωy)], (N,N); n_shift = [N/2,N/2])

# This linear map is constructed to map a N × N image into `length(Ωx)` k-space samples.
# So its
# [adjoint](https://en.wikipedia.org/wiki/Adjoint)
# goes the other direction.
# However, an adjoint is *not* an inverse!

gridded2 = A' * vec(data)
jim(x, y, gridded2, title="NUFFT gridding without DCF")


# ## Density compensation

# To get a decent image with NUFFT-based gridding of non-Cartesian data,
# one must compensate for the k-space sampling density.
# See
# [this book chapter](https://web.eecs.umich.edu/~fessler/book/c-four.pdf)
# for details.

# Because this example is using radial sampling,
# we can borrow ideas from tomography,
# especially the ramp filter,
# to define a reasonable density compensation function (DCF).

# Here is a basic DCF version that uses the ramp filter in a simple way.
# The `dcf .* data` line uses Julia's broadcast feature
# to apply the 1D DCF to each radial spoke.

dcf = abs.(kr) # (N+1) weights along k-space polar coordinate 
dcf = dcf / oneunit(eltype(dcf)) # kludge: units not working with LinearMap now
gridded3 = A' * vec(dcf .* data)
p3 = jim(x, y, gridded3, title="NUFFT gridding with simple ramp-filter DCF")


# The image is more reasonable than without any DCF,
# but we can do better using the correction of
# [Lauzon&Rutt, 1996](http://doi.org/10.1002/mrm.1910360617)
# and
# [Joseph, 1998](http://doi.org/10.1002/mrm.1910400317).

dν = 1/FOV # k-space radial sample spacing
dcf = pi / Nϕ * dν * abs.(kr) # see lauzon:96:eop, joseph:98:sei
dcf[kr .== 0] .= pi * (dν/2)^2 / Nϕ # area of center disk
dcf = dcf / oneunit(eltype(dcf)) # kludge: units not working with LinearMap now
gridded4 = A' * vec(dcf .* data)
p4 = jim(x, y, gridded4, title="NUFFT gridding with better ramp-filter DCF")


# Finally we have a NUFFT gridded image with DCF
# that has the appropriate range of values,
# but it still looks less than ideal.
# So next we try an iterative approach.


# ## Iterative MR image reconstruction using an NUFFT.

# todo


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg
Pkg.status()
