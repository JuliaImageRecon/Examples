#---------------------------------------------------------
# # [Compressed Sensing 2D pMRI ](@id 4-cs-sense-2d)
#---------------------------------------------------------

#=
This example illustrates how to perform
2D compressed sensing image reconstruction
from Cartesian sampled MRI data
for parallel MRI (sensitivity encoding)
with 1-norm regularization of orthogonal wavelet coefficients,
using the Julia language.

This page was generated using a single Julia file:
[4-cs-sense-2d.jl](@__REPO_ROOT_URL__/mri/4-cs-sense-2d.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`4-cs-sense-2d.ipynb`](@__NBVIEWER_ROOT_URL__/mri/4-cs-sense-2d.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`4-cs-sense-2d.ipynb`](@__BINDER_ROOT_URL__/mri/4-cs-sense-2d.ipynb).

#=
This demo is somewhat similar to Fig. 3 in the survey paper
"[Optimization methods for MR image reconstruction](http://doi.org/10.1109/MSP.2019.2943645),"
in Jan 2020 IEEE Signal Processing Magazine,
except
* the sampling is 1D phase encoding instead of 2D,
* there are multiple coils,
* we use units # todo
* the simulation avoids inverse crimes.
=#

# Packages used in this demo (run `Pkg.add` as needed):
using ImagePhantoms: ellipse_parameters, SheppLoganBrainWeb, ellipse, phantom
using ImagePhantoms: mri_smap_fit, mri_spectra
#using ImageFiltering: imfilter, centered
using ImageMorphology: dilate #, label_components # imfill
using LazyGrids: ndgrid
using ImageGeoms: embed, embed!
using MIRT: Aodwt # todo Asense
using MIRTjim: jim, prompt
using MIRT: ir_mri_sensemap_sim
using MIRT: pogm_restart
using LinearAlgebra: norm, dot
using LinearMapsAA: LinearMapAA
using Plots; default(markerstrokecolor=:auto, label="")
using FFTW: fft!, bfft!, fftshift!
using Random: seed!
using Unitful: mm
using InteractiveUtils: versioninfo

if !@isdefined(ydata) || true # false # todo remove

# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


# ### Create (synthetic) data

# Image geometry:

fovs = (256, 256) .* 1mm # todo: units!
nx, ny = (192, 256)
dx, dy = fovs ./ (nx,ny)
x = (-(nx÷2):(nx÷2-1)) * dx
y = (-(ny÷2):(ny÷2-1)) * dy


# Modified Shepp-Logan phantom with random complex phases per ellipse
object = ellipse_parameters(SheppLoganBrainWeb() ; disjoint=true, fovs)
seed!(0)
object = vcat( (object[1][1:end-1]..., 1), # random phases
    [(ob[1:end-1]..., randn(ComplexF32)) for ob in object[2:end]]...)
# object = ellipse(object)
object = ellipse([object[1]]) # todo: simplify
oversample = 3
Xtrue = phantom(x, y, object, oversample)
cfun = z -> cat(dims = ndims(z)+1, real(z), imag(z))
#clim = (0,9)
jim(:aspect_ratio, :equal)
jim(x, y, cfun(Xtrue), "True image\n real | imag"; ncol=2)


# Mask (support for image reconstruction)

mask = abs.(Xtrue) .> 0
mask = dilate(dilate(dilate(mask))) # dilate twice with 3×3 square
@assert mask .* Xtrue == Xtrue
jim(x, y, mask + abs.(Xtrue), "Mask + |Xtrue|")


# Make sensitivity maps, normalized so SSoS = 1:
ncoil = 4
smap_raw = ir_mri_sensemap_sim(dims=(nx,ny); ncoil, orbit_start=[45])
jif(args...; kwargs...) = jim(args...; prompt=false, kwargs...)
p1 = jif(x, y, smap_raw, "Sensitivity maps raw");

sum_last = (f, x) -> selectdim(sum(f, x; dims=ndims(x)), ndims(x), 1)
ssos_fun = smap -> sqrt.(sum_last(abs2, smap)) # SSoS
ssos_raw = ssos_fun(smap_raw) # SSoS raw
p2 = jif(x, y, ssos_raw, "SSoS raw, ncoil=$ncoil");

smap = @. smap_raw / ssos_raw * mask # normalize and mask
ssos = ssos_fun(smap) # SSoS
@assert all(≈(1), @view ssos[mask]) # verify ≈ 1
smaps = [eachslice(smap; dims = ndims(smap))...] # stack
p3 = jif(x, y, smaps, "|Sensitivity maps normalized|")
p4 = jif(x, y, map(x -> angle.(x), smaps), "∠Sensitivity maps"; color=:hsv)
jim(p1, p2, p3, p4)


# Frequency sample vectors:
fx = (-(nx÷2):(nx÷2-1)) / (nx*dx) # crucial to match `mri_smap_basis` internals!
fy = (-(ny÷2):(ny÷2-1)) / (ny*dy)
gx, gy = ndgrid(fx, fy);

# Somewhat random 1D phase-encode sampling:
seed!(0); sampfrac = 0.4; samp = rand(ny÷2) .< sampfrac
tmp = rand(ny÷2) .< 0.5; samp = [samp .* tmp; reverse(samp .* .!tmp)] # symmetry
samp .|= (abs.(fy*dy) .< 1/8) # fully sampled center ±1/8 phase-encodes
samp = trues(ny) # todo: temp full sampling
ny_count = count(samp)
samp = trues(nx) * samp'
samp_frac = round(100*count(samp) / (nx*ny), digits=2)
jim(fx, fy, samp, title="k-space sampling ($ny_count / $ny = $samp_frac%)")
#src jim(samp + 1*reverse(samp), fft0=true) # check symmetry


# To avoid an inverse crime, here we use the 2012 method of
# [Guerquin-Kern et al.](http://doi.org/10.1109/TMI.2011.2174158)
# and use the analytical k-space values of the phantom
# combined with an analytical model for the sensitivity maps.

kmax = 7
fit = mri_smap_fit(smaps, embed; mask, kmax, deltas=(dx,dy))
jim(
 jif(x, y, cfun(smaps), "Original maps"; clim=(-1,1), nrow=4),
 jif(x, y, cfun(fit.smaps), "Fit maps"; clim=(-1,1), nrow=4),
 jif(x, y, cfun(100 * (fit.smaps - smaps)), "error * 100"; nrow=4),
 layout = (1,3),
)

# Analytical spectra computation for complex phantom using all smaps
# (no inverse crime here):
ytrue = mri_spectra(gx[samp], gy[samp], object, fit)
ytrue = hcat(ytrue...)

# Noisy under-sampled k-space data:
sig = 1
sig = 0 # todo test
ydata = ytrue + sig * √(2) * randn(ComplexF32, size(ytrue)) # complex noise!
ydata = ComplexF32.(ydata) # save memory
ysnr = 20 * log10(norm(ytrue) / norm(ydata - ytrue)) # data SNR in dB

# Display zero-filled data:
logger = (x; min=-6, up=maximum(abs,x)) -> log10.(max.(abs.(x) / up, (10.)^min))
jim(:abswarn, false) # suppress warnings about showing magnitude
tmp = embed(ytrue[:,1],samp)
jim(
 jif(fx, fy, logger(tmp),
    title="k-space |data| (zero-filled, coil 1)",
    xlabel="νx", ylabel="νy"),
 jif(fx, fy, angle.(tmp),
    title="∠data, coil 1"; color=:hsv)
)

end # ydata


#=
### Prepare to reconstruct
Creating a system matrix (encoding matrix) and an initial image.

The system matrix is a `LinearMapAA` object,
akin to a `fatrix` in Matlab MIRT.
=#

# System model ("encoding matrix") for 2D image `x` being mapped
# to array of size `count(samp) × ncoil` k-space data

# todo: this needs a lot of work.
# should have high-level "Asense" that can wrap around Afft or Anufft
# each of which having its own in-place work, with tests.

function Asense(samp, smaps)
    dims = size(smaps[1])
    N = prod(dims)
    work1 = Array{ComplexF32}(undef, dims)
    work2 = Array{ComplexF32}(undef, dims)
    ncoil = length(smaps)
    function forw!(y, x)
        for ic in 1:ncoil
            @. work1 = x * smaps[ic]
            fftshift!(work1, fft!(fftshift!(work2, work1)))
            y[:,ic] .= work1[samp]
        end
    end
    function back!(x, y)
        for ic in 1:ncoil
            embed!(work1, (@view y[:,ic]), samp)
#           bfft!(work2, work1)
            fftshift!(work1, bfft!(fftshift!(work2, work1)))
            copyto!(work2, smaps[ic])
            conj!(work2)
            if ic == 1
                @. x = work1 * work2
            else
                @. x += work1 * work2
            end
        end
    end
    A = LinearMapAA(forw!, back!, (ncoil*count(samp), N);
        odim = (count(samp),ncoil), idim=dims, T=ComplexF32)
    return A
end

#=
The `dx * dy` factor here is required
because the true k-space data `ytrue`
comes from an analytical Fourier transform
but the reconstruction uses a discrete Fourier transform.
=#
A = Asense(samp, smaps) * (dx * dy) # operator!

# validate adjoint
if false
    tmp1 = randn(ComplexF32, A._idim)
    tmp2 = randn(ComplexF32, A._odim)
    @assert isapprox(dot(tmp2, A * tmp1), dot(A' * tmp2, tmp1); rtol=1e-4)
end

# Compare the analytical k-space data with the discrete modeled k-space data
y0 = embed(ytrue, samp)
y1 = embed(A * Xtrue, samp)
jim(
 jif(logger(y0; up=maximum(abs,y0)), "analytical"; clim=(-6,0)),
 jif(logger(y1; up=maximum(abs,y0)), "discrete"; clim=(-6,0)),
 jif(logger(y1 - y0; up=maximum(abs,y0)), "difference"),
)
# norm(y1) / norm(y0) # scale factor is ≈1

# Initial image based on zero-filled reconstruction.
# Note the `dx*dy` scale factor here!
nrmse = (x) -> round(norm(x - Xtrue) / norm(Xtrue) * 100, digits=1)
X0 = 1.0f0/(nx*ny) * (A' * ydata) / (dx*dy)^2
jim(x, y, X0, "|X0|: initial image; NRMSE $(nrmse(X0))%")
throw()


#=
## Wavelet sparsity in synthesis form

The image reconstruction optimization problem here is
```math
\arg \min_{x}
\frac{1}{2} \| A x - y \|_2^2 + \beta \; \| W x \|_1
```
where
``y`` is the k-space data,
``A`` is the system model (simply Fourier encoding `F` here),
``W`` is an orthogonal discrete (Haar) wavelet transform,
again implemented as a `LinearMapAA` object.
Because ``W`` is unitary,
we make the change of variables
``z = W x``
and solve for ``z``
and then let ``x = W' z``
at the end.
In fact we use a weighted 1-norm
where only the detail wavelet coefficients are regularized,
not the approximation coefficients.
=#

# Orthogonal discrete wavelet transform operator (`LinearMapAO`):
W, scales, _ = Aodwt((nx,ny) ; T = eltype(A))
isdetail = scales .> 0
jim(
 jif(scales, "wavelet scales"),
 jif(real(W * Xtrue) .* isdetail, "wavelet detail coefficients"),
)


# Inputs needed for proximal gradient methods:
Az = A * W' # another operator!
Fnullz = (z) -> 0 # cost function in `z` not needed
f_gradz = (z) -> Az' * (Az * z - ydata)
f_Lz = nx*ny # Lipschitz constant for single coil Cartesian DFT
regz = 0.03 * nx * ny # oracle from Xtrue wavelet coefficients!
costz = (z) -> 1/2 * norm(Az * z - ydata)^2 + regz * norm(z,1) # 1-norm regularizer
soft = (z,c) -> sign(z) * max(abs(z) - c, 0) # soft thresholding
g_prox = (z,c) -> soft.(z, isdetail .* (regz * c)) # proximal operator (shrink details only)
z0 = W * X0
jim(z0, "Initial wavelet coefficients")


#=
## Iterate

Run ISTA=PGM and FISTA=FPGM and POGM, the latter two with adaptive restart
See [Kim & Fessler, 2018](http://doi.org/10.1007/s10957-018-1287-4)
for adaptive restart algorithm details.
=#

# Functions for tracking progress:
function fun_ista(iter, xk_z, yk, is_restart)
    xh = W' * xk_z
    return (costz(xk_z), nrmse(xh), is_restart) # , psnr(xh) # time()
end

function fun_fista(iter, xk, yk_z, is_restart)
    xh = W' * yk_z
    return (costz(yk_z), nrmse(xh), is_restart) # , psnr(xh) # time()
end;

# Run and compare three proximal gradient methods:
niter = 20
z_ista, out_ista = pogm_restart(z0, Fnullz, f_gradz, f_Lz;
    mom=:pgm, niter,
    restart=:none, restart_cutoff=0., g_prox, fun=fun_ista)
Xista = W'*z_ista
@show nrmse(Xista)

z_fista, out_fista = pogm_restart(z0, Fnullz, f_gradz, f_Lz;
    mom=:fpgm, niter,
    restart=:gr, restart_cutoff=0., g_prox, fun=fun_fista)
Xfista = W'*z_fista
@show nrmse(Xfista)

z_pogm, out_pogm = pogm_restart(z0, Fnullz, f_gradz, f_Lz;
    mom=:pogm, niter,
    restart=:gr, restart_cutoff=0., g_prox, fun=fun_fista)
Xpogm = W'*z_pogm
@show nrmse(Xpogm)

jim(
 jif(x, y, Xfista, "FISTA/FPGM"),
 jif(x, y, Xpogm, "POGM with ODWT"),
)
#src savefig("xpogm_odwt.pdf")


# ## POGM is fastest

# Plot cost function vs iteration:
cost_ista = [out_ista[k][1] for k in 1:niter+1]
cost_fista = [out_fista[k][1] for k in 1:niter+1]
cost_pogm = [out_pogm[k][1] for k in 1:niter+1]
cost_min = min(minimum(cost_ista), minimum(cost_pogm))
pc = plot(xlabel="iteration k", ylabel="Relative cost")
scatter!(0:niter, cost_ista  .- cost_min, label="Cost ISTA")
scatter!(0:niter, cost_fista .- cost_min, markershape=:square, label="Cost FISTA")
scatter!(0:niter, cost_pogm  .- cost_min, markershape=:utriangle, label="Cost POGM")
#src savefig("cost_pogm_odwt.pdf")

#
isinteractive() && prompt();


# Plot NRMSE vs iteration:
nrmse_ista = [out_ista[k][2] for k in 1:niter+1]
nrmse_fista = [out_fista[k][2] for k in 1:niter+1]
nrmse_pogm = [out_pogm[k][2] for k in 1:niter+1]
pn = plot(xlabel="iteration k", ylabel="NRMSE %")#, ylims=(3,6.5))
scatter!(0:niter, nrmse_ista, label="NRMSE ISTA")
scatter!(0:niter, nrmse_fista, markershape=:square, label="NRMSE FISTA")
scatter!(0:niter, nrmse_pogm, markershape=:utriangle, label="NRMSE POGM")
#src savefig("nrmse_pogm_odwt.pdf")

#
isinteractive() && prompt();

# Show error images:
p1 = jif(x, y, Xtrue, "true")
p2 = jif(x, y, X0, "X0: initial"; xlabel = "$(nrmse(X0))%")
p3 = jif(x, y, Xpogm, "POGM recon"; xlabel = "$(nrmse(Xpogm))%")
p5 = jif(x, y, X0 - Xtrue, "X0 error", clim=(0,2))
p6 = jif(x, y, Xpogm - Xtrue, "Xpogm error", clim=(0,2))
pe = jim(p2, p3, p5, p6)



# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions:

import Pkg; Pkg.status()
