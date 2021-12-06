#---------------------------------------------------------
# # [NUFFT Overview](@id 1-nufft)
#---------------------------------------------------------

#=
This example illustrates how to use Nonuniform FFT (NUFFT)
for image reconstruction in MRI
using the Julia language.

This entire page was generated using a single Julia file:
[1-nufft.jl](@__REPO_ROOT_URL__/mri/1-nufft.jl).
=#
#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`1-nufft.ipynb`](@__NBVIEWER_ROOT_URL__/mri/1-nufft.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`1-nufft.ipynb`](@__BINDER_ROOT_URL__/mri/1-nufft.ipynb).

#=
Some MRI scans use non-Cartesian sampling patterns
(like radial and spiral k-space trajectories, among others),
often in the interest of acquisition speed.

Image reconstruction from fully sampled Cartesian k-space data
typically uses simple inverse FFT operations,
whereas non-Cartesian sampling requires more complicated methods.

The examples here illustrate both non-iterative (aka "gridding")
and iterative methods for non-Cartesian MRI reconstruction.
For simplicity the examples consider the case of single-coil data
and ignore the effects of B0 field inhomogeneity.


First we add the Julia packages that are need for these examples.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

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

using ImagePhantoms: shepp_logan, SheppLoganEmis, spectrum, phantom #, Gauss2
using Plots; default(label="", markerstrokecolor=:auto)
using Unitful: mm # Allows use of physical units (mm here)
using UnitfulRecipes # Ensures that plots show appropriate units when applicable
using LaTeXStrings # for LaTeX in plot labels, e.g., L"\alpha_n"
using MIRTjim: jim, prompt # jiffy image display
using MIRT: Anufft, diffl_map, ncg
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


# ## Radial k-space sampling

#=
We focus on radial sampling as a simple representative non-Cartesian case.
Consider imaging a 256mm × 256mm field of FOV
with the goal of reconstructing a 128 × 128 pixel image.
The following radial and angular k-space sampling is reasonable.
=#

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


#=
For the NUFFT routines considered here,
the sampling arguments must be "Ω" values:
digital frequencies have pseudo-units of radians / pixel.
=#

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
isinteractive() && prompt();


# ## Radial k-space data for Shepp-Logan phantom

#=
Get the ellipse parameters for a MRI-suitable version of the Shepp-Logan phantom
and calculate (analytically) the radial k-space data.
Then display in polar coordinates.
=#

object = shepp_logan(SheppLoganEmis(); fovs=(FOV,FOV))
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

#=
It would be impossible for a radiologist
to diagnose a patient
from the k-space data in polar coordinates,
so image reconstruction is needed.

The simplest approach
is to (nearest-neighbor) interpolate the k-space data
onto a Cartesian grid,
and then reconstruction with an inverse FFT.

One way to do that interpolation
is to use a `Histogram` method
in Julia's statistics package.
=#

using StatsBase: fit, Histogram, weights

#=
The following function is a work-around because `weights` in StatsBase is
[limited to Real data](https://github.com/JuliaStats/StatsBase.jl/issues/745),
so here we bin the real and imaginary k-space data separately,
and handle the units.
=#

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
clim = (0, 9)
p0 = jim(x, y, ideal, title="True Shepp-Logan phantom"; clim)


# ## NUFFT approach to gridding

#=
Basic nearest-neighbor gridding
does not provide acceptable image quality in MRI,
so now we turn to using the NUFFT.
For MRI purposes,
the NUFFT is a function that maps Cartesian spaced image data
into non-Cartesian k-space data.
The
[NFFT.jl](https://github.com/tknopp/NFFT.jl)
package has functions
for computing the NUFFT and its adjoint.
These are linear mappings (generalizations of matrices),
so instead of calling those functions directly,
here we use the NUFFT linear map object
defined in MIRT
that provides a non-Cartesian Fourier encoding "matrix".
=#

A = Anufft([vec(Ωx) vec(Ωy)], (N,N); n_shift = [N/2,N/2])

#=
This linear map is constructed to map a N × N image into `length(Ωx)` k-space samples.
So its
[adjoint](https://en.wikipedia.org/wiki/Adjoint)
goes the other direction.
However, an adjoint is *not* an inverse!
=#

gridded2 = A' * vec(data)
jim(x, y, gridded2, title="NUFFT gridding without DCF")


# ## Density compensation

#=
To get a decent image with NUFFT-based gridding of non-Cartesian data,
one must compensate for the k-space sampling density.
See
[this book chapter](https://web.eecs.umich.edu/~fessler/book/c-four.pdf)
for details.

Because this example uses radial sampling,
we can borrow ideas from tomography,
especially the
[ramp filter](https://en.wikipedia.org/wiki/Radon_transform#Radon_inversion_formula),
to define a reasonable density compensation function (DCF).

Here is a basic DCF version that uses the ramp filter in a simple way,
corresponding to the areas of annular segments
(Voronoi cells in polar coordinates).
The `dcf .* data` line uses Julia's
[broadcast](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting)
feature
to apply the 1D DCF to each radial spoke.
=#

dν = 1/FOV # k-space radial sample spacing
dcf = pi / Nϕ * dν * abs.(kr) # (N+1) weights along k-space polar coordinate
dcf = dcf / oneunit(eltype(dcf)) # kludge: units not working with LinearMap now
gridded3 = A' * vec(dcf .* data)
p3 = jim(x, y, gridded3, title="NUFFT gridding with simple ramp-filter DCF"; clim)


#=
The image is more reasonable than without any DCF,
but we can do better (quantitatively) using the correction of
[Lauzon&Rutt, 1996](http://doi.org/10.1002/mrm.1910360617)
and
[Joseph, 1998](http://doi.org/10.1002/mrm.1910400317).
=#

dcf = pi / Nϕ * dν * abs.(kr) # see lauzon:96:eop, joseph:98:sei
dcf[kr .== 0/mm] .= pi * (dν/2)^2 / Nϕ # area of center disk
dcf = dcf / oneunit(eltype(dcf)) # kludge: units not working with LinearMap now
gridded4 = A' * vec(dcf .* data)
p4 = jim(x, y, gridded4, title="NUFFT gridding with better ramp-filter DCF"; clim)


# A profile helps illustrate the improvement.

pp = plot(x, real(gridded4[:,N÷2]), label="Modified ramp DCF")
plot!(x, real(gridded3[:,N÷2]), label="Basic ramp DCF")
plot!(x, real(ideal[:,N÷2]), label="Ideal", xlabel=L"x", ylabel="middle profile")

#
isinteractive() && prompt();


#=
Finally we have made a NUFFT gridded image with DCF
that has the appropriate range of values,
but it still looks less than ideal.
So next we try an iterative approach.


## Iterative MR image reconstruction using NUFFT

As an initial iterative approach,
let's apply a few iterations of conjugate gradient (CG)
to seek the minimizer of the least-squares cost function:
```math
\arg \min_{x} \frac{1}{2} \| A x - y \|_2^2.
```
CG is well-suited to minimizing quadratic cost functions,
but we do not expect the image to be great quality
because radial sampling omits the "corners" of k-space
so the NUFFT operator ``A`` is badly conditioned.
=#

dx = FOV / N # pixel size
dx = dx / oneunit(dx) # abandon units for now
gradf = u -> u - vec(data/dx^2) # gradient of f(u) = 1/2 \| u - y \|^2
curvf = u -> 1 # curvature of f(u)
x0 = gridded4 # initial guess: best gridding reconstruction
xls, _ = ncg([A], [gradf], [curvf], x0; niter = 20)
p5 = jim(x, y, xls, "LS-CG reconstruction"; clim)


#=
To improve the results, we include regularization.
Here we would like to reconstruct an image
by finding the minimizer of a regularized LS cost function
such as the following:
```math
\arg \min_{x} \frac{1}{2} \| A x - y \|_2^2 + \beta R(x)
, \qquad
R(x) = 1' \psi.(T x).
```
We focus here on the case of edge-preserving regularization
where ``T`` is a 2D finite-differencing operator
and ``\psi`` is a potential function.
This operator maps a N×N image into a N×N×2 array
with the horizontal and vertical finite differences.
=#

T = diffl_map((N,N), [1,2] ; T = ComplexF32)


# Applying this operator to the ideal image illustrated its action:
jim(x, y, T * ideal; ncol=1, title="Horizontal and vertical finite differences")


#=
## Edge-preserving regularization

We use the Fair potential function:
a rounded corner version of absolute value,
an approximation to anisotropic total variation (TV).
=#

β = 2^13 # regularization parameter
δ = 0.05 # edge-preserving parameter
wpot = z -> 1 / (1 + abs(z)/δ) # weighting function


#=
## Nonlinear CG algorithm

We apply a (nonlinear) CG algorithm
to seek the minimizer of the cost function.
Nonlinear CG is well suited to convex problems
that are locally quadratic
like the regularized cost function considered here.
See
[this survey paper](http://doi.org/10.1109/MSP.2019.2943645)
for an overview of optimization methods for MRI.
=#

B = [A, T] # see MIRT.ncg
gradf = [u -> u - vec(data/dx^2), # data-term gradient, correct for pixel area
         u -> β * (u .* wpot.(u))] # regularizer gradient
curvf = [u -> 1, u -> β] # curvature of quadratic majorizers
x0 = gridded4 # initial guess is best gridding reconstruction
xhat, _ = ncg(B, gradf, curvf, x0; niter = 90)
p6 = jim(x, y, xhat, "Iterative reconstruction"; clim)


# Here is a comparison of the profiles.

plot!(pp, x, real(xls[:,N÷2]), label="LS-CG")
plot!(pp, x, real(xhat[:,N÷2]), label="Iterative edge-preserving", color=:black)

#
isinteractive() && prompt();

#=
In this case, iterative image reconstruction provides the best looking image.
One might argue this simulation was doomed to succeed,
because the phantom is piece-wise constant,
which is the best case for edge-preserving regularization.
On the other hand,
this was not an
[inverse crime](http://doi.org/10.1016/j.cam.2005.09.027)
([see also here](http://arxiv.org/abs/2109.08237))
because the k-space data came from the analytical spectrum of ellipses,
rather than from a discrete image.


# ### Caveats

* The phantom used here was real-valued, which is unrealistic
  (although the reconstruction methods did not "know" it was real).
* This simulation is for a single-coil scan
  whereas modern MRI scanners generally use multiple receive coils.
* There was no statistical noise in this simulation.
=#


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
