#=
# [DCF-based "preconditioning" in MRI](@id 6-nufft)

This example illustrates
how "preconditioning" based on sampling
density compensation factors (DCFs)
affects image reconstruction in MRI
using the Julia language.
=#

#srcURL

#=
The bottom line here
is that DCF-based preconditioning
gives an apparent speed-up
when staring from a zero initial image,
but leads to increased noise (worse error).
Choosing a smart starting image,
like an appropriately scaled gridding image,
provides comparable speed-up
without compromising noise.
=#


#=
First we add the Julia packages that are need for these examples.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "InteractiveUtils"
        "ImagePhantoms"
        "LaTeXStrings"
        "MIRTjim"
        "MIRT"
        "Plots"
        "Unitful"
    ])
end


# Now tell this Julia session to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ImagePhantoms: shepp_logan, SheppLoganEmis, spectrum, phantom #, Gauss2
using LaTeXStrings
using LinearAlgebra: I, norm
using MIRTjim: jim, prompt # jiffy image display
using MIRT: Anufft, diffl_map, ncg
using Plots; default(label="", markerstrokecolor=:auto)
using Random: seed!; seed!(0)
using Unitful: mm # physical units (mm here)
using InteractiveUtils: versioninfo


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


#=
## Radial k-space sampling

We consider radial sampling
as a simple representative non-Cartesian case.
Consider imaging a 256mm × 256mm field of FOV
with the goal of reconstructing a 128 × 128 pixel image.
The following radial and angular k-space sampling is reasonable.
=#

N = 128
FOV = 256mm # physical units!
Δx = FOV / N # pixel size
kmax = 1 / 2Δx
kr = ((-N÷2):(N÷2)) / (N÷2) * kmax # radial sampling in k-space
Nr = length(kr) # N+1
Nϕ = N-2 # theoretically should be about π/2 * N
kϕ = (0:Nϕ-1)/Nϕ * π # angular samples
νx = kr * cos.(kϕ)' # N × Nϕ k-space sampling in cycles/mm
νy = kr * sin.(kϕ)'
psamp = plot(νx, νy, size = (420,400),
    xaxis = (L"\nu_{\mathrm{x}}", (-1,1) .* (1.1 * kmax), (-1:1) * kmax),
    yaxis = (L"\nu_{\mathrm{y}}", (-1,1) .* (1.1 * kmax), (-1:1) * kmax),
    aspect_ratio = 1,
    title = "Radial k-space sampling",
)

#
isinteractive() && prompt();
#src savefig(psamp, "sampling.pdf")


#=
For the NUFFT routines considered here,
the sampling arguments must be "Ω" values:
digital frequencies have pseudo-units of radians / pixel.
=#

Ωx = (2π * Δx) * νx # N × Nϕ grid of k-space sample locations
Ωy = (2π * Δx) * νy # in pseudo-units of radians / sample

pom = scatter(Ωx, Ωy,
    xaxis = (L"\Omega_x", (-1,1) .* 1.1π, ((-1:1)*π, ["-π", "0", "π"])),
    yaxis = (L"\Omega_y", (-1,1) .* 1.1π, ((-1:1)*π, ["-π", "0", "π"])),
    aspect_ratio = 1, markersize = 0.5,
    title = "Radial k-space sampling",
)

#
isinteractive() && prompt();


#=
## Radial k-space data for Shepp-Logan phantom

Get the ellipse parameters for a MRI-suitable version of the Shepp-Logan phantom
and calculate (analytically) the radial k-space data.
Then display in polar coordinates.
=#

object = shepp_logan(SheppLoganEmis(); fovs=(FOV,FOV))
#src object = [Gauss2(18mm, 0mm, 100mm, 70mm, 0, 1)] # useful for validating DCF
data = spectrum(object).(νx,νy)
data = data / oneunit(eltype(data)) # abandon units at this point
dscale = 10000
jimk = (args...; kwargs...) -> jim(kr, kϕ, args...;
    xaxis = (L"k_r", (-1,1) .* kmax, (-1:1) .* maximum(abs, kr)),
    yaxis = (L"k_{\phi}", (0,π), (0,π)),
    aspect_ratio = :none,
    kwargs...
)
pk = jimk(abs.(data) / dscale; title="k-space data magnitude / $dscale")


#=
Here is what the phantom should look like ideally.
=#

x = (-(N÷2):(N÷2-1)) * Δx
y = (-(N÷2):(N÷2-1)) * Δx
ideal = phantom(x, y, object, 2)
clim = (0, 9)
p0 = jim(x, y, ideal; xlabel=L"x", ylabel=L"y", clim, size = (460,400),
 title="True Shepp-Logan phantom")

#
isinteractive() && prompt();
#src savefig(p0, "ideal.pdf")


#=
## NUFFT operator
with sanity check
=#

A = Anufft([vec(Ωx) vec(Ωy)], (N,N); n_shift = [N/2,N/2]) # todo: odim=(Nr,Nϕ)
dx = FOV / N # pixel size
dx = dx / oneunit(dx) # abandon units for now
Ax_to_y = Ax -> dx^2 * reshape(Ax, Nr, Nϕ) # trick
pj1 = jimk(abs.(Ax_to_y(A * ideal)) / dscale, "|A*x|/$dscale")
pj2 = jimk(abs.(Ax_to_y(A * ideal) - data) / dscale, "|A*x-y|/$dscale")
plot(pk, pj1, pj2)

#
isinteractive() && prompt();


#=
## Density compensation

Radial sampling needs
(N+1) DSF weights along k-space polar coordinate.

We use the improved method of
[Lauzon&Rutt, 1996](https://doi.org/10.1002/mrm.1910360617)
and
[Joseph, 1998](https://doi.org/10.1002/mrm.1910400317).
=#

dν = 1/FOV # k-space radial sample spacing
dcf = π / Nϕ * dν * abs.(kr) # see lauzon:96:eop, joseph:98:sei
dcf[kr .== 0/mm] .= π * (dν/2)^2 / Nϕ # area of center disk
dcf = dcf / oneunit(eltype(dcf)) # kludge: units not working with LinearMap now
gridded4 = A' * vec(dcf .* data)
p4 = jim(x, y, gridded4; xlabel=L"x", ylabel=L"y", clim,
 size = (460,400),
 title="NUFFT gridding with better ramp-filter DCF")

#
isinteractive() && prompt();
#src savefig(p4, "gridding.pdf")


# A profile shows it is "decent" but not amazing.

pp = plot(x, real(gridded4[:,N÷2]), label="Modified ramp DCF")
plot!(x, real(ideal[:,N÷2]), label="Ideal", xlabel=L"x", ylabel="middle profile")

#
isinteractive() && prompt();


#=
## Unregularized iterative MRI reconstruction

Apply a few iterations of conjugate gradient (CG)
to approach a minimizer of the least-squares cost function:
```math
\arg \min_{x} \frac{1}{2} ‖ A x - y ‖₂².
```

Using a zero-image as the starting point ``x₀``
leads to slow convergence.

Using "preconditioning" by including DCF values
in the cost function
(a kind of weighted LS)
appears to give much faster convergence,
but leads to suboptimal image quality
especially in the presence of noise.

Using a decent initial image
(e.g., a gridded image with appropriate scaling)
is preferable.

There is a subtle point here about `dx`
when converting the Fourier integral to a sum.
Here `yideal` is `data/dx^2`.
=#

nrmse = x -> norm(x - ideal) / norm(ideal) * 100
fun = (x, iter) -> nrmse(x)

snr2sigma(db, y) = 10^(-db/20) * norm(y) / sqrt(length(y))
yideal = vec(data/dx^2)
σnoise = snr2sigma(40, yideal)
ydata = yideal + σnoise * randn(eltype(yideal), size(yideal))
snr = 20 * log10(norm(yideal) / norm(ydata - yideal)) # check
gradf = u -> u - ydata # gradient of f(u) = 1/2 ‖ u - y ‖²
curvf = u -> 1 # curvature of f(u)

x0 = 0 * ideal
niter = 15
xls0, out0 = ncg([A], [gradf], [curvf], x0; niter, fun)

default(linewidth = 2)
ppr = plot(
 xaxis = ("iteration", (0, niter)),
 yaxis = ("NRMSE", (0, 100), 0:20:100),
 widen = true,
)
plot!(ppr, 0:niter, out0,
 label = "Non-preconditioned, 0 init", color = :red, marker = :circle,
);

# "Preconditioned" version
precon = vec(repeat(dcf, Nϕ));

# gradient of fp(u) = 1/2 ‖ P^{1/2} (u - y) ‖²
gradfp = u -> precon .* (u - ydata)
curvfp = u -> precon # curvature of fp(u)

xlsp, out2 = ncg([A], [gradfp], [curvfp], x0; niter, fun)
plot!(ppr, 0:niter, out2,
 label = "DCF preconditioned, 0 init", color = :blue, marker = :x,
)
#src savefig(ppr, "precon0.pdf")

# Smart start with gridded image
x0 = gridded4 # initial guess: decent gridding reconstruction
xls1, out1 = ncg([A], [gradf], [curvf], x0; niter, fun)
plot!(ppr, 0:niter, out1,
  label = "Non-preconditioned, gridding init", color=:green, marker=:+,
)

#src savefig(ppr, "precon1.pdf")

#
isinteractive() && prompt();

# The images look very similar at 15 iterations

elim = (0, 1)
tmp = stack((ideal, xls0, xlsp, xls1))
err = stack((ideal, xls0, xlsp, xls1)) .- ideal
p5 = plot(
 jim(x, y, tmp; title = "Ideal | 0 init | precon | grid init",
  clim, nrow=1, size = (600, 200)),
 jim(x, y, err; title = "Error", color=:cividis,
  clim = elim, nrow=1, size = (600, 200)),
 layout = (2, 1),
 size = (650, 450),
)

#
isinteractive() && prompt();


include("../../inc/reproduce.jl")
