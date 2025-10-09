#=
# [L+S 2D dynamic recon](@id 5-l-plus-s)

This page illustrates dynamic parallel MRI image reconstruction
using a low-rank plus sparse (L+S) model
optimized by a fast algorithm
described in the paper
by Claire Lin and Jeff Fessler
[Efficient Dynamic Parallel MRI Reconstruction for the Low-Rank Plus Sparse Model](https://doi.org/10.1109/TCI.2018.2882089),
IEEE Trans. on Computational Imaging, 5(1):17-26, 2019,
by Claire Lin and Jeff Fessler,
EECS Department, University of Michigan.

The Julia code here is a translation
of part of the
[Matlab code](https://github.com/JeffFessler/reproduce-l-s-dynamic-mri)
used in the original paper.

If you use this code,
please cite that paper.
=#

#srcURL

# ### Setup

# Packages needed here.

## using Unitful: s
using Plots; cgrad, default(markerstrokecolor=:auto, label="")
using MIRT: Afft, Asense, embed
using MIRT: pogm_restart, poweriter
using MIRTjim: jim, prompt
using FFTW: fft!, bfft!, fftshift!
using LinearMapsAA: LinearMapAA, block_diag, redim, undim
using MAT: matread
import Downloads # todo: use Fetch or DataDeps?
using LinearAlgebra: dot, norm, svd, svdvals, Diagonal, I
using Random: seed!
using Statistics: mean
using LaTeXStrings


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

jif(args...; kwargs...) = jim(args...; prompt=false, kwargs...)
isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Overview

Dynamic image reconstruction
using a "low-rank plus sparse"
or "L+S" approach
was proposed by
[Otazo et al.](https://doi.org/10.1002/mrm.25240)
and uses the following cost function:

```math
X = \hat{L} + \hat{S}
,\qquad
(\hat{L}, \hat{S})
= \arg \min_{L,S} \frac{1}{2} \| E (L + S) - d \|_2^2
 + λ_L \| L \|_*
 + λ_S \| vec(T S) \|_1
```
where ``T`` is a temporal unitary FFT,
``E`` is an encoding operator (system matrix),
and ``d``
is Cartesian undersampled multicoil k-space data.

The Otazo paper used an
iterative soft thresholding algorithm (ISTA)
to solve this optimization problem.
Using FISTA is faster,
but using
the
[proximal optimized gradient method (POGM)](https://doi.org/10.1137/16m108104x)
with
[adaptive restart](https://doi.org/10.1007/s10957-018-1287-4)
is even faster.

This example reproduces part of Figures 1 & 2 in
[Claire Lin's paper](https://doi.org/10.1109/TCI.2018.2882089),
based on the
[cardiac perfusion example](https://github.com/JeffFessler/reproduce-l-s-dynamic-mri/blob/master/examples/example_cardiac_perf.m).
=#

# ## Read data
if !@isdefined(data)
#src url = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/"
    url = "https://github.com/JeffFessler/MIRTdata/raw/main/mri/lin-19-edp/"
    dataurl = url * "cardiac_perf_R8.mat"
    data = matread(Downloads.download(dataurl))
    xinfurl = url * "Xinf.mat"
    Xinf = matread(Downloads.download(xinfurl))["Xinf"]["perf"] # (128,128,40)
end;

# Show converged image as a preview:
pinf = jim(Xinf, L"\mathrm{Converged\ image\ sequence } X_∞")

# Organize k-space data:
if !@isdefined(ydata0)
    ydata0 = data["kdata"] # k-space data full of zeros
    ydata0 = permutedims(ydata0, [1, 2, 4, 3]) # (nx,ny,nc,nt)
    ydata0 = ComplexF32.(ydata0)
end
(nx, ny, nc, nt) = size(ydata0)


# Extract sampling pattern from zeros of k-space data:
if !@isdefined(samp)
    samp = ydata0[:,:,1,:] .!= 0
    for ic in 2:nc # verify it is same for all coils
        @assert samp == (ydata0[:,:,ic,:] .!= 0)
    end
    kx = -(nx÷2):(nx÷2-1)
    ky = -(ny÷2):(ny÷2-1)
    psamp = jim(kx, ky, samp, "Sampling patterns for $nt frames";
       xlabel=L"k_x", ylabel=L"k_y")
end

#=
Are all k-space rows are sampled in one of the 40 frames?
Sadly no.
The 10 blue rows shown below are never sampled.
A better sampling pattern design
could have avoided this issue.
=#
samp_sum = sum(samp, dims=3)
color = cgrad([:blue, :black, :white], [0, 1/2nt, 1])
pssum = jim(kx, ky, samp_sum; xlabel="kx", ylabel="ky",
    color, clim=(0,nt), title="Number of sampled frames out of $nt")

# Prepare coil sensitivity maps
if !@isdefined(smaps)
    smaps_raw = data["b1"] # raw coil sensitivity maps
    jim(smaps_raw, "Raw |coil maps| for $nc coils")
    sum_last = (f, x) -> selectdim(sum(f, x; dims=ndims(x)), ndims(x), 1)
    ssos_fun = smap -> sqrt.(sum_last(abs2, smap)) # SSoS
    ssos_raw = ssos_fun(smaps_raw)
    smaps = smaps_raw ./ ssos_raw
    ssos = ssos_fun(smaps)
    @assert all(≈(1), ssos)
    pmap = jim(smaps, "Normalized |coil maps| for $nc coils")
end


#=
Temporal unitary FFT sparsifying transform
for image sequence of size `(nx, ny, nt)`:
=#
TF = Afft((nx,ny,nt), 3; unitary=true) # unitary FFT along 3rd (time) dimension
if false # verify adjoint
    tmp1 = randn(ComplexF32, nx, ny, nt)
    tmp2 = randn(ComplexF32, nx, ny, nt)
    @assert dot(tmp2, TF * tmp1) ≈ dot(TF' * tmp2, tmp1)
    @assert TF' * (TF * tmp1) ≈ tmp1
    (size(TF), TF._odim, TF._idim)
end


#=
Examine temporal Fourier sparsity of Xinf.
The low temporal frequencies dominate,
as expected,
because Xinf was reconstructed
using this regularizer!
=#
tmp = TF * Xinf
ptfft = jim(tmp, "|Temporal FFT of Xinf|")


#=
## System matrix
Construct dynamic parallel MRI system model.
It is block diagonal
where each frame has its own sampling pattern.
The input (image) here has size `(nx=128, ny=128, nt=40)`.
The output (data) has size `(nsamp=2048, nc=12, nt=40)`
because every frame
has 16 phase-encode lines of 128 samples.

todo: precompute (i)fft along readout direction to save time

The code in the original Otazo et al. paper
used an `ifft` in the forward model
and an `fft` in the adjoint,
so we must use a flag here to match that model.
=#
Aotazo = (samp, smaps) -> Asense(samp, smaps; unitary=true, fft_forward=false) # Otazo style
A = block_diag([Aotazo(s, smaps) for s in eachslice(samp, dims=3)]...)
#A = ComplexF32(1/sqrt(nx*ny)) * A # match Otazo's scaling
(size(A), A._odim, A._idim)

#src check forward model
#src tmp = A * Xinf
#src tmp2 = [embed(tmp[:,:,it], samp[:,:,it]) for it in 1:nt]
#src tmp = cat(dims=4, tmp2...)

# Reshape data to match the system model
if !@isdefined(ydata)
    tmp = reshape(ydata0, :, nc, nt)
    tmp = [tmp[vec(s),:,it] for (it,s) in enumerate(eachslice(samp, dims=3))]
    ydata = cat(tmp..., dims=3) # (nsamp,nc,nt) = (2048,12,40) no "zeros"
end
size(ydata)


# Final encoding operator `E` for L+S because we stack `X = [L;S]`
tmp = LinearMapAA(I(nx*ny*nt);
    odim=(nx,ny,nt), idim=(nx,ny,nt), T=ComplexF32, prop=(;name="I"))
tmp = kron([1 1], tmp)
AII = redim(tmp; odim=(nx,ny,nt), idim=(nx,ny,nt,2)) # "squeeze" odim
E = A * AII;

# Run power iteration to verify that `opnorm(E) = √2`
if false
    (_, σ1E) = poweriter(undim(E)) # 1.413 ≈ √2
else
    σ1E = √2
end

# Check scale factor of Xinf. (It should be ≈1.)
tmp = A * Xinf
scale0 = dot(tmp, ydata) / norm(tmp)^2 # 1.009 ≈ 1

# Crude initial image
L0 = A' * ydata # adjoint (zero-filled)
#src # no optimized scaling in Lin 2019 paper
#src tmp = A * L0
#src L0 .*= dot(tmp, ydata) / norm(tmp)^2 # optimal initial scaling
S0 = zeros(ComplexF32, nx, ny, nt)
X0 = cat(L0, S0, dims=ndims(L0)+1) # (nx, ny, nt, 2) = (128, 128, 40, 2)
M0 = AII * X0 # L0 + S0
pm0 = jim(M0, "|Initial L+S via zero-filled recon|")


#=
## L+S reconstruction
Prepare for proximal gradient methods
=#

# Scalars to match Otazo's results
scaleL = 130 / 1.2775 # Otazo's stopping St(1) / b1 constant squared
scaleS = 1 / 1.2775; # 1 / b1 constant squared

# L+S regularizer
lambda_L = 0.01 # regularization parameter
lambda_S = 0.01 * scaleS
Lpart = X -> selectdim(X, ndims(X), 1) # extract "L" from X
Spart = X -> selectdim(X, ndims(X), 2) # extract "S" from X
nucnorm(L::AbstractMatrix) = sum(svdvals(L)) # nuclear norm
nucnorm(L::AbstractArray) = nucnorm(reshape(L, :, nt)); # (nx*ny, nt) for L

# Optimization overall composite cost function
Fcost = X -> 0.5 * norm(E * X - ydata)^2 +
    lambda_L * scaleL * nucnorm(Lpart(X)) + # note scaleL !
    lambda_S * norm(TF * Spart(X), 1);

f_grad = X -> E' * (E * X - ydata); # gradient of data-fit term

#=
Lipschitz constant of data-fit term is 2
because A is unitary and AII is like ones(2,2).
=#
f_L = 2; # σ1E^2

# Proximal operator for scaled nuclear norm ``β | X |_*``:
# singular value soft thresholding (SVST).
function SVST(X::AbstractArray, β)
    dims = size(X)
    X = reshape(X, :, dims[end]) # assume time frame is the last dimension
    U,s,V = svd(X)
    sthresh = @. max(s - β, 0)
    keep = findall(>(0), sthresh)
    X = U[:,keep] * Diagonal(sthresh[keep]) * V[:,keep]'
    X = reshape(X, dims)
    return X
end;

# Combine proximal operators for L and S parts to make overall prox for `X`
soft = (v,c) -> sign(v) * max(abs(v) - c, 0) # soft threshold function
S_prox = (S, β) -> TF' * soft.(TF * S, β) # 1-norm proximal mapping for unitary TF
g_prox = (X, c) -> cat(dims=ndims(X),
    SVST(Lpart(X), c * lambda_L * scaleL),
    S_prox(Spart(X), c * lambda_S),
);

if false # check functions
    @assert Fcost(X0) isa Real
    tmp = f_grad(X0)
    @assert size(tmp) == size(X0)
    tmp = SVST(Lpart(X0), 1)
    @assert size(tmp) == size(L0)
    tmp = S_prox(S0, 1)
    @assert size(tmp) == size(S0)
    tmp = g_prox(X0, 1)
    @assert size(tmp) == size(X0)
end


niter = 10
fun = (iter, xk, yk, is_restart) -> (Fcost(xk), xk); # logger

# Run PGM
if !@isdefined(Mpgm)
    f_mu = 2/0.99 - f_L # trick to match 0.99 step size in Lin 1999
    f_mu = 0
    xpgm, out_pgm = pogm_restart(X0, (x) -> 0, f_grad, f_L ;
        f_mu, mom = :pgm, niter, g_prox, fun)
    Mpgm = AII * xpgm
end;

# Run FPGM (FISTA)
if !@isdefined(Mfpgm)
    xfpgm, out_fpgm = pogm_restart(X0, (x) -> 0, f_grad, f_L ;
        mom = :fpgm, niter, g_prox, fun)
    Mfpgm = AII * xfpgm
end;

# Run POGM
if !@isdefined(Mpogm)
    xpogm, out_pogm = pogm_restart(X0, (x) -> 0, f_grad, f_L ;
        mom = :pogm, niter, g_prox, fun)
    Mpogm = AII * xpogm
end;

# Look at final POGM image components
px = jim(
 jif(Lpart(xpogm), "L"),
 jif(Spart(xpogm), "S"),
 jif(Mpogm, "M=L+S"),
 jif(Xinf, "Minf"),
)

# Plot cost function
costs = out -> [o[1] for o in out]
nrmsd = out -> [norm(AII*o[2]-Xinf)/norm(Xinf) for o in out]
cost_pgm = costs(out_pgm)
cost_fpgm = costs(out_fpgm)
cost_pogm = costs(out_pogm)
pc = plot(xlabel = "iteration", ylabel = "cost")
plot!(0:niter, cost_pgm, marker=:circle, label="PGM (ISTA)")
plot!(0:niter, cost_fpgm, marker=:square, label="FPGM (FISTA)")
plot!(0:niter, cost_pogm, marker=:star, label="POGM")

# Plot NRMSD vs Matlab Xinf
nrmsd_pgm = nrmsd(out_pgm)
nrmsd_fpgm = nrmsd(out_fpgm)
nrmsd_pogm = nrmsd(out_pogm)
pd = plot(xlabel = "iteration", ylabel = "NRMSD vs Matlab Xinf")
plot!(0:niter, nrmsd_pgm, marker=:circle, label="PGM (ISTA)")
plot!(0:niter, nrmsd_fpgm, marker=:square, label="FPGM (FISTA)")
plot!(0:niter, nrmsd_pogm, marker=:star, label="POGM")

#src # todo: need fully sampled data like in Fig2 of paper and in OnAir to proceed

#src tmpfile = "/Users/fessler/dat/git/mine/reproduce-l-s-dynamic-mri/examples/tmp.mat"
#src debug = matread(tmpfile)
#src @assert debug["tmp"] ≈ M0

#src ix=33:96; iy=33:96; it=[2,8,14,24] # frames

#src Function for computing RMSE within the mask
#src frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)) * s, digits=1) / s;


#src Run each algorithm twice; once to track rmse and costs, once for timing
#src function runner

#src Compare final RMSE values
#src frmse.((ftrue, finit, fmap_cg_n, fmap_cg_d, fmap_cg_c, fmap_cg_i))

#src Plot NRMSE vs wall time

#=
## Discussion

todo
=#

include("../../inc/reproduce.jl")
