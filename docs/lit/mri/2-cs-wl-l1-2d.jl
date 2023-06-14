#=
# [Compressed Sensing 2D](@id 2-cs-wl-l1-2d)

This example illustrates how to perform
2D compressed sensing image reconstruction
from Cartesian sampled MRI data
with 1-norm regularization of orthogonal wavelet coefficients,
using the Julia language.
=#

#srcURL

#=
This demo is somewhat similar to Fig. 3 in the survey paper
"[Optimization methods for MR image reconstruction](https://doi.org/10.1109/MSP.2019.2943645),"
in Jan 2020 IEEE Signal Processing Magazine,
except that the sampling is 1D phase encoding instead of 2D.
=#

# Packages used in this demo (run `Pkg.add` as needed):
using ImagePhantoms: shepp_logan, SheppLoganEmis, spectrum, phantom
using MIRT: embed, Afft, Aodwt
using MIRTjim: jim, prompt
using MIRT: pogm_restart
using LinearAlgebra: norm
using Plots; default(markerstrokecolor=:auto, label="")
using FFTW: fft
using Random: seed!
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


# ## Create (synthetic) data

# Shepp-Logan phantom (unrealistic because real-valued):
nx,ny = 192,256
object = shepp_logan(SheppLoganEmis(); fovs=(ny,ny))
Xtrue = phantom(-(nx÷2):(nx÷2-1), -(ny÷2):(ny÷2-1), object, 2)
Xtrue = reverse(Xtrue, dims=2)
clim = (0,9)
jim(Xtrue, "true image"; clim)
#src savefig("xtrue.pdf")


# Somewhat random 1D phase-encode sampling:
seed!(0); sampfrac = 0.2; samp = rand(ny) .< sampfrac; sig = 1
mod2 = (N) -> mod.((0:N-1) .+ Int(N/2), N) .- Int(N/2)
samp .|= (abs.(mod2(ny)) .< Int(ny/8)) # fully sampled center rows
samp = trues(nx) * samp'
jim(samp, fft0=true, title="k-space sampling ($(100count(samp)/(nx*ny))%)")
#src savefig(p1, "samp.pdf")

# Generate noisy, under-sampled k-space data (inverse-crime simulation):
ytrue = fft(Xtrue)[samp]
y = ytrue + sig * √(2) * randn(ComplexF32, size(ytrue)) # complex noise!
y = ComplexF32.(y) # save memory
ysnr = 20 * log10(norm(ytrue) / norm(y-ytrue)) # data SNR in dB

# Display zero-filled data:
logger = (x; min=-6) -> log10.(max.(abs.(x) / maximum(abs, x), (10.)^min))
jim(:abswarn, false) # suppress warnings about showing magnitude
jim(logger(embed(ytrue,samp)), fft0=true, title="k-space |data| (zero-filled)",
    xlabel="kx", ylabel="ky")


#=
## Prepare to reconstruct
Creating a system matrix (encoding matrix) and an initial image  
The system matrix is a `LinearMapAA` object, akin to a `fatrix` in Matlab MIRT.
=#

# System model ("encoding matrix") from MIRT:
F = Afft(samp) # operator!

# Initial image based on zero-filled reconstruction:
nrmse = (x) -> round(norm(x - Xtrue) / norm(Xtrue) * 100, digits=1)
X0 = 1.0f0/(nx*ny) * (F' * y)
jim(X0, "|X0|: initial image; NRMSE $(nrmse(X0))%")


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
W, scales, _ = Aodwt((nx,ny) ; T = eltype(F))
isdetail = scales .> 0
jim(
    jim(scales, "wavelet scales"),
    jim(real(W * Xtrue) .* isdetail, "wavelet detail coefficients"),
)


# Inputs needed for proximal gradient methods:
Az = F * W' # another operator!
Fnullz = (z) -> 0 # cost function in `z` not needed
f_gradz = (z) -> Az' * (Az * z - y)
f_Lz = nx*ny # Lipschitz constant for single coil Cartesian DFT
regz = 0.03 * nx * ny # oracle from Xtrue wavelet coefficients!
costz = (z) -> 1/2 * norm(Az * z - y)^2 + regz * norm(z,1) # 1-norm regularizer
soft = (z,c) -> sign(z) * max(abs(z) - c, 0) # soft thresholding
g_prox = (z,c) -> soft.(z, isdetail .* (regz * c)) # proximal operator (shrink details only)
z0 = W * X0
jim(z0, "Initial wavelet coefficients")


#=
## Iterate

Run ISTA=PGM and FISTA=FPGM and POGM, the latter two with adaptive restart
See [Kim & Fessler, 2018](https://doi.org/10.1007/s10957-018-1287-4)
for adaptive restart algorithm details.
=#

# Functions for tracking progress:
function fun_ista(iter, xk_z, yk, is_restart)
    xh = W' * xk_z
    return (costz(xk_z), nrmse(xh), is_restart) # , psnr(xh)) # time()
end

function fun_fista(iter, xk, yk_z, is_restart)
    xh = W' * yk_z
    return (costz(yk_z), nrmse(xh), is_restart) # , psnr(xh)) # time()
end;

# Run and compare three proximal gradient methods:
niter = 50
z_ista, out_ista = pogm_restart(z0, Fnullz, f_gradz, f_Lz; mom=:pgm, niter=niter,
    restart=:none, restart_cutoff=0., g_prox=g_prox, fun=fun_ista)
Xista = W'*z_ista
@show nrmse(Xista)

z_fista, out_fista = pogm_restart(z0, Fnullz, f_gradz, f_Lz; mom=:fpgm, niter=niter,
    restart=:gr, restart_cutoff=0., g_prox=g_prox, fun=fun_fista)
Xfista = W'*z_fista
@show nrmse(Xfista)

z_pogm, out_pogm = pogm_restart(z0, Fnullz, f_gradz, f_Lz; mom=:pogm, niter=niter,
    restart=:gr, restart_cutoff=0., g_prox=g_prox, fun=fun_fista)
Xpogm = W'*z_pogm
@show nrmse(Xpogm)

jim(
    jim(Xfista, "FISTA/FPGM"),
    jim(Xpogm, "POGM with ODWT"),
)
#src savefig("xpogm_odwt.pdf")


# ## POGM is fastest

# Plot cost function vs iteration:
cost_ista = [out_ista[k][1] for k=1:niter+1]
cost_fista = [out_fista[k][1] for k=1:niter+1]
cost_pogm = [out_pogm[k][1] for k=1:niter+1]
cost_min = min(minimum(cost_ista), minimum(cost_pogm))
plot(xlabel="iteration k", ylabel="Relative cost")
scatter!(0:niter, cost_ista  .- cost_min, label="Cost ISTA")
scatter!(0:niter, cost_fista .- cost_min, markershape=:square, label="Cost FISTA")
scatter!(0:niter, cost_pogm  .- cost_min, markershape=:utriangle, label="Cost POGM")
#src savefig("cost_pogm_odwt.pdf")

#
isinteractive() && prompt();


# Plot NRMSE vs iteration:
nrmse_ista = [out_ista[k][2] for k=1:niter+1]
nrmse_fista = [out_fista[k][2] for k=1:niter+1]
nrmse_pogm = [out_pogm[k][2] for k=1:niter+1]
pn = plot(xlabel="iteration k", ylabel="NRMSE %", ylims=(3,6.5))
scatter!(0:niter, nrmse_ista, label="NRMSE ISTA")
scatter!(0:niter, nrmse_fista, markershape=:square, label="NRMSE FISTA")
scatter!(0:niter, nrmse_pogm, markershape=:utriangle, label="NRMSE POGM")
#src savefig("nrmse_pogm_odwt.pdf")

# Show error images:
p1 = jim(Xtrue, "true")
p2 = jim(X0, "X0: initial")
p3 = jim(Xpogm, "POGM recon")
p5 = jim(X0 - Xtrue, "X0 error", clim=(0,2))
p6 = jim(Xpogm - Xtrue, "Xpogm error", clim=(0,2))
pe = jim(p2, p3, p5, p6)


include("../../inc/reproduce.jl")
