#---------------------------------------------------------
# # [B0 field map](@id 5-l-plus-s)
#---------------------------------------------------------

#=
This page illustrates dynamic parallel MRI image reconstruction
using a low-rank plus sparse (L+S) model
as illustrated in the paper
by Claire Lin and Jeff Fessler
https://github.com/JeffFessler/reproduce-l-s-dynamic-mri
"Efficient Dynamic Parallel MRI Reconstruction
for the Low-Rank Plus Sparse Model,
"IEEE Trans. on Computational Imaging, 5(1):17-26, 2019,
by Claire Lin and Jeff Fessler,
EECS Department, University of Michigan
[http://doi.org/10.1109/TCI.2018.2882089].

The Julia code here is a translation
of the 
[Matlab code used in the original paper](https://github.com/JeffFessler/reproduce-l-s-dynamic-mri/blob/master/README.md).

If you use this code,
please cite that paper.

This page was generated from a single Julia file:
[5-l-plus-s.jl](@__REPO_ROOT_URL__/5-l-plus-s.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`5-l-plus-s.ipynb`](@__NBVIEWER_ROOT_URL__/5-l-plus-s.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`5-l-plus-s.ipynb`](@__BINDER_ROOT_URL__/5-l-plus-s.ipynb).


# ### Setup

# Packages needed here.

#using Unitful: s
using Plots; default(markerstrokecolor=:auto, label="")
using MIRT: Afft, Asense
using MIRT: pogm_restart, poweriter
using MIRTjim: jim, prompt
using FFTW: fft!, bfft!, fftshift!
using LinearMapsAA: LinearMapAA, block_diag, redim, undim
using MAT: matread
import Downloads # todo: use Fetch or DataDeps?
using LinearAlgebra: dot, norm, I
using Random: seed!
using StatsBase: mean
using LaTeXStrings

if !@isdefined(ydata) # todo

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

jif(args...; kwargs...) = jim(args...; prompt=false, kwargs...)
isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
### Overview

Dynamic image reconstruction
using a "low-rank plus sparse"
or "L+S" approach
was proposed by
[Otazo et al.](http://doi.org/10.1002/mrm.25240)
and uses the following cost function:

```math
X = \hat{L} + \hat{S}
,\qquad
(\hat{L}, \hat{S})
= \arg \min_{L,S} \frac{1}{2} \| A (L + S) - d \|_2^2
 + λ_L \| L \|_*
 + λ_S \| T S \|_1
```
where ``T`` is a temporal unitary FFT
and ``d``
is Cartesian undersampled multicoil k-space data.

The Otazo paper used an
iterative soft thresholding algorithm (ISTA)
to solve this optimization problem.
Using FISTA is faster,
but using
the
[proximal optimized gradient method (POGM)](http://doi.org/10.1137/16m108104x)
with
[adaptive restart](http://doi.org/10.1007/s10957-018-1287-4)
is even faster.

This example reproduces part of Figures 1 & 2 in
[Claire Lin's paper](http://doi.org/10.1109/TCI.2018.2882089),
based on the
[cardiac perfusion example](https://github.com/JeffFessler/reproduce-l-s-dynamic-mri/blob/master/examples/example_cardiac_perf.m).
=#

# ## Read data
if !@isdefined(data)
    url = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/"
    dataurl = url * "cardiac_perf_R8.mat"
    data = matread(Downloads.download(dataurl))
    xinfurl = url * "Xinf.mat"
    Xinf = matread(Downloads.download(xinfurl))["Xinf"]["perf"] # (128,128,40)
end;

# Show converged image:
jim(Xinf, L"\mathrm{Converged\ image\ } X_∞")

# Organize k-space data:
ydata0 = data["kdata"] # k-space data full of zeros
ydata0 = permutedims(ydata0, [1, 2, 4, 3]) # (nx,ny,nc,nt)
(nx,ny,nc,nt) = size(ydata0)

# Extract sampling pattern from zeros of k-space data:
samp = ydata0[:,:,1,:] .!= 0
for ic in 2:nc # verify it is same for all coils
    @assert samp == (ydata0[:,:,ic,:] .!= 0)
end
kx = -(nx÷2):(nx÷2-1)
ky = -(ny÷2):(ny÷2-1)
jim(kx, ky, samp, "Samping patterns for $nt frames"; xlabel=L"k_x", ylabel=L"k_y")

# Prepare coil sensitivity maps
smaps_raw = data["b1"] # raw coil sensitivity maps
jim(smaps_raw, "Raw |coil maps| for $nc coils")
sum_last = (f, x) -> selectdim(sum(f, x; dims=ndims(x)), ndims(x), 1)
ssos_fun = smap -> sqrt.(sum_last(abs2, smap)) # SSoS
ssos_raw = ssos_fun(smaps_raw)
smaps = smaps_raw ./ ssos_raw
ssos = ssos_fun(smaps)
@assert all(≈(1), ssos)
jim(smaps, "Normalized |coil maps| for $nc coils")

#=
Temporal unitary FFT sparsifying transform
for image sequence of size (nx, ny, nt):
=#
function makeTF(dims::Dims, nt::Int; T = ComplexF32)
    N = prod(dims)
    function forw!(y, x)
        fft!(copyto!(y, x), length(dims)+1) # FFT along time dimension
        y ./= sqrt(N) # unitary
    end
    function back!(x, y)
        bfft!(copyto!(x, y), length(dims)+1) # iFFT along time dimension
        x ./= sqrt(N) # unitary
    end
    A = LinearMapAA(forw!, back!, prod(dims)*nt .* (1,1);
        odim = (dims..., nt), idim = (dims..., nt), T)
    return A
end
TF = makeTF((nx,ny), nt)
if false # verify adjoint
    tmp1 = randn(ComplexF32, nx, ny, nt)
    tmp2 = randn(ComplexF32, nx, ny, nt)
    @assert dot(tmp2, TF * tmp1) ≈ dot(TF' * tmp2, tmp1)
end
(size(TF), TF._odim, TF._idim)

#=
Examine temporal Fourier sparsity of Xinf.
The low temporal frequencies dominate,
as expected,
because Xinf was reconstructed
using this regularizer!
=#
tmp = TF * Xinf
jim(tmp, "|Temporal FFT of Xinf|")

#=
## System matrix
Construct dynamic parallel MRI system model.
It is block diagonal
where each frame has its own sampling pattern.
The input (image) here has size `(nx=128, ny=128, nt=40)`.
The output (data) has size `(nsamp=2048, nc=12, nt=40)`
because every frame
has 16 phase-encode lines of 128 samples.
=#
A = block_diag([Asense(s, smaps) for s in eachslice(samp, dims=3)]...)
A = ComplexF32(1/sqrt(nx*ny)) * A # match Otazo's scaling
(size(A), A._odim, A._idim)

# Reshape data to match the system model
tmp = reshape(ydata0, :, nc, nt)
tmp = [tmp[vec(s),:,it] for (it,s) in enumerate(eachslice(samp, dims=3))]
ydata = cat(tmp..., dims=3) # (nsamp,nc,nt) = (2048,12,40) no "zeros"
size(ydata)

end # ydata

# Final encoding operator for L+S because we stack [L;S]
tmp = LinearMapAA(I(nx*ny*nt);
    odim=(nx,ny,nt), idim=(nx,ny,nt), T=ComplexF32, prop=(;name="I"))
tmp = kron([1 1], tmp)
AII = redim(tmp; odim=(nx,ny,nt), idim=(nx,ny,nt,2)) # "squeeze" odim
E = A * AII

# Run power iteration to verify that `opnorm(E) = √2`
if false
    (_, σ1) = poweriter(undim(E))
end

# Check scale factor of Xinf. (It should be ≈1.)
tmp = A * Xinf
scale = dot(tmp, ydata) / norm(tmp)^2 # todo: why not ≈1?

# Crude initial image
L0 = A' * ydata # adjoint (zero-filled)
tmp = A * L0
L0 .*= dot(tmp, ydata) / norm(tmp)^2 # optimal initial scaling
S0 = zeros(nx, ny, nt)
X0 = cat(L0, S0, dims=ndims(L0)+1) # (nx, ny, nt, 2) = (128, 128, 40, 2)
M0 = AII * X0 # L0 + S0
jim(M0, "Initial L+S via zero-filled rcon")


#=
## L+S reconstruction
=#

# scalars to match Otazo's results
scaleL = 130 / 1.2775 # Otazo's stopping St(1) / b1 constant squared
scaleS = 1 / 1.2775 # 1 / b1 constant squared

#=
%% prepare for AL: opt
niter = 10
opt.muL=0.01;
opt.muS=0.01*opt.scaleS;
opt.Xinf = Xinf.perf;
%% AL-CG
d1 = 1/5; d2 = 1/5; %for AL-CG
[L_cg,S_cg,x_cg,cost_cg,time_cg,rankL_cg] = AL_CG(opt,'d1',d1,'d2',d2);
%% AL-2
d1 = 1/5; d2 = 1/50; %for AL-2
[L_al,S_al,xdiff_al,cost_al,time_al,rankL_al] = AL_2(opt,'d1',d1,'d2',d2);
=#

#=
Prepare for proximal gradient methods

Lipschitz constant of data-fit term is 2
because A is unitary and AII is like ones(2,2).
=#
f_L = 2

# Proximal operator for scaled nuclear norm ``β | X |_*``:
# singular value soft thresholding (SVST).
nucnorm(L::AbstractMatrix) = sum(svdvals(L)) # nuclear norm
nucnorm(L::AbstractArray) = nucnorm(reshape(L, :, nt)) # (nx*ny, nt) for L
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

f_grad = X -> E' * (E * X - ydata) # gradient of data-fit term

# L+S regularizer
lambda_L = 0.01 # regularization parameter
lambda_S = 0.01 * scaleS
Lpart = X -> selectdim(X, ndims(X), 1) # extract "L" from X
Spart = X -> selectdim(X, ndims(X), 2) # extract "S" from X
Fcost = X -> 0.5 * norm(E * X - ydata)^2 +
    lambda_L * nucnorm(Lpart(X)) +
    lambda_S * norm(TF * Spart(X)) # optimization cost function

# L and S proximal operators
soft = (v,c) -> sign(v) * max(abs(v) - c, 0) # soft threshold function
S_prox = (S, β) -> TF' * soft.(TF * S, β) # 1-norm proximal mapping for unitary TF
g_prox = (X, c) -> cat(dims=ndims(X),
    SVST(Lpart(X), c * lambda_L),
    S_prox(Spart(X), c * lambda_S),
)

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
fun = (iter, xk, yk, is_restart) -> Fcost(xk)

xpogm, out_pogm = pogm_restart(X0, (x) -> 0, f_grad, f_L ;
    mom = :pogm, niter, g_prox, fun)
Mpogm = AII * xpogm

px = jim(
 jif(Lpart(xpogm), "L"),
 jif(Spart(xpogm), "S"),
 jif(Mpogm, "M=L+S"),
 jif(Xinf, "Minf"),
)

pc = plot(xlabel = "iteration", ylabel = "cost")
plot!(0:niter, out_pogm, marker=:star, label="POGM")

throw()


# ISTA
# [L_ista,S_ista,xdiff_ista,cost_ista,time_ista,rankL_ista] = PGM(param);
# FISTA
# [L_fista,S_fista,xdiff_fista,cost_fista,time_fista,rankL_fista] = PGM(param,'fistaL',1,'fistaS',1);
# POGM
# [L_pogm,S_pogm,xdiff_pogm,cost_pogm,time_pogm,rankL_pogm] = PGM(param,'pogmS',1,'pogmL',1);

#=
%% Display: 4 frames
L = L_pogm;S = S_pogm;
LplusS=L+S;
LplusSd=LplusS(33:96,33:96,2);LplusSd=cat(2,LplusSd,LplusS(33:96,33:96,8));LplusSd=cat(2,LplusSd,LplusS(33:96,33:96,14));LplusSd=cat(2,LplusSd,LplusS(33:96,33:96,24));
Ld=L(33:96,33:96,2);Ld=cat(2,Ld,L(33:96,33:96,8));Ld=cat(2,Ld,L(33:96,33:96,14));Ld=cat(2,Ld,L(33:96,33:96,24));
Sd=S(33:96,33:96,2);Sd=cat(2,Sd,S(33:96,33:96,8));Sd=cat(2,Sd,S(33:96,33:96,14));Sd=cat(2,Sd,S(33:96,33:96,24));
figure;
subplot(3,1,1),imshow(abs(LplusSd),[0,1]);ylabel('L+S')
subplot(3,1,2),imshow(abs(Ld),[0,.03]);ylabel('L')
subplot(3,1,3),imshow(abs(Sd),[0,1]);ylabel('S')
=#

# Extract arrays used in simulation

# Function for computing RMSE within the mask
# frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)) * s, digits=1) / s;


#=
## Run NCG

Run each algorithm twice; once to track rmse and costs, once for timing
=#
yik_scale = ydata / scale
fmap_run = (niter, precon, track; kwargs...) ->
    b0map(yik_scale, echotime; smap, mask,
       order=1, l2b=-4, gamma_type=:PR, niter, precon, track, kwargs...)

function runner(niter, precon; kwargs...)
    (fmap, _, out) = fmap_run(niter, precon, true; kwargs...) # tracking run
    (_, times, _) = fmap_run(niter, precon, false; kwargs...) # timing run
    return (fmap, out.fhats, out.costs, times)
end;


# ### 2. NCG: no precon
if !@isdefined(fmap_cg_n)
    niter_cg_n = 50
    (fmap_cg_n, fhat_cg_n, cost_cg_n, time_cg_n) = runner(niter_cg_n, :I)

    pcost = plot(time_cg_n, cost_cg_n, marker=:circle, label="NCG-MLS");
    pi_cn = jim(fmap_cg_n, "CG:I"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_n)) Hz")
end


# ### 3. NCG: diagonal preconditioner
if !@isdefined(fmap_cg_d)
    niter_cg_d = 40
    (fmap_cg_d, fhat_cg_d, cost_cg_d, time_cg_d) = runner(niter_cg_d, :diag)

    plot!(pcost, time_cg_d, cost_cg_d, marker=:square, label="NCG-MLS-D")
    pi_cd = jim(fmap_cg_d, "CG:diag"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_d)) Hz")
end


# ### 4. NCG: Cholesky preconditioner
# (This one may use too much memory for larger images.)
if !@isdefined(fmap_cg_c)
    niter_cg_c = 3
    (fmap_cg_c, fhat_cg_c, cost_cg_c, time_cg_c) = runner(niter_cg_c, :chol)

    plot!(pcost, time_cg_c, cost_cg_c, marker=:square, label="NCG-MLS-C")
    pi_cc = jim(fmap_cg_c, "CG:chol"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_c)) Hz")
end


# ### 5. NCG: Incomplete Cholesky preconditioner
if !@isdefined(fmap_cg_i)
    niter_cg_i = 14
    (fmap_cg_i, fhat_cg_i, cost_cg_i, time_cg_i) =
        runner(niter_cg_i, :ichol; lldl_args = (; memory=20, droptol=0))

    plot!(pcost, time_cg_i, cost_cg_i, marker=:square, label="NCG-MLS-IC",
        xlabel = "time [s]", ylabel="cost")
    pi_ci = jim(fmap_cg_i, "CG:ichol"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_i)) Hz")
end


# Compare final RMSE values
frmse.((ftrue, finit, fmap_cg_n, fmap_cg_d, fmap_cg_c, fmap_cg_i))

# Plot RMSE vs wall time
prmse = plot(xlabel = "time [s]", ylabel="RMSE [Hz]")
fun = (time, fhat, label) ->
    plot!(prmse, time, frmse.(eachslice(fhat; dims=4)); label, marker=:circ)
fun(time_cg_n, fhat_cg_n, "None")
fun(time_cg_d, fhat_cg_d, "Diag")
fun(time_cg_c, fhat_cg_c, "Chol")
fun(time_cg_i, fhat_cg_i, "IC")

#=
## Discussion

That final figure is similar to Fig. 4 of the 2020 Lin&Fessler paper,
after correcting that figure for a
[factor of π](https://github.com/ClaireYLin/regularized-field-map-estimation).

This figure was generated in github's cloud,
where the servers are busily multi-tasking,
so the compute times per iteration
can vary widely between iterations and runs.

Nevertheless,
it is interesting that
in this Julia implementation
the diagonal preconditioner
seems to be
as effective as the incomplete Cholesky preconditioner.
=#
