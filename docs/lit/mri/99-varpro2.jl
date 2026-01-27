#=
# [Variable Projection: Two exponentials](@id varpro2)

Illustrate fitting bi-exponential model to data.

See:
- [VarPro blog](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals)
- [VP4Optim.jl](https://github.com/cganter/VP4Optim.jl) has biexponential fits
- [Varpro.jl](https://github.com/macd/Varpro.jl)

=#

#srcURL

# ### Setup

# Packages needed here.

import ForwardDiff
using ImagePhantoms: ellipse, ellipse_parameters, phantom, SheppLoganBrainWeb
using Statistics: mean, std
using Plots: default, gui, histogram, plot, plot!, scatter, scatter!
using Plots: cgrad, RGB
default(markerstrokecolor=:auto, label="", widen = true)
using LaTeXStrings
using LinearAlgebra: norm, Diagonal, diag, diagm, qr
using MIRTjim: jim
using Random: seed!; seed!(0)
#using Unitful: @u_str, ms, s, mm
ms = 0.001; s = 1; mm = 1 # avoid Unitful due to ForwardDiff
using InteractiveUtils: versioninfo


#=
## Double exponential (bi-exponential)

We explore a simple case:
fitting a bi-exponential to some noisy data:
```math
y_m = c_a e^{- r_a t_m} + c_b e^{- r_b t_m} + ϵ_m
,\quad
m = 1,…,M.
```
The four unknown parameters here are:
- the decay rates ``r_a, r_b ≥ 0``
- the amplitudes (aka coefficients) ``c_a, c_b``
  (that could be complex in some MRI settings)
=#

Tf = Float32
Tc = Complex{Tf}
M = 18 # how many samples (more than in single exponential demo!)
Δte = 10ms # echo spacing
te1 = 5ms # time of first echo
tm = Tf.(te1 .+ (0:(M-1)) * Δte) # echo times
Lin = (:ca, :cb) # linear model parameter names
Non = (:ra, :rb) # nonlinear model parameter names
Tlin = NamedTuple{Lin}
Tnon = NamedTuple{Non}
Tall = NamedTuple{(Lin..., Non...)}
c_true = Tlin(Tf.([60, 40])) # AU
r_true = Tnon(Tf.([100/s, 20/s]))
x_true = (; c_true..., r_true...) # all parameters

#=
## Signal model

This next function is the
signal basis function(s) from physics.
This is the only function that is model-specific.
=#
signal_bases(ra::Number, rb::Number, t::Number) =
    [exp(-t * ra); exp(-t * rb)]; # bi-exponential model

#=
The following signal helper functions
apply to many models
having a mix of linear and nonlinear signal parameters.

Here there is just one scan parameter (echo time).

- These functions would need to be generalized
to handle multiple scan parameters
(e.g., echo time, phase cycling factor, flip angle).

- They would also need to be generalized
to handle models with "known" parameters
(e.g., B0 and B1+).
=#
signal_bases(non::Tnon, t::Number) =
    signal_bases(non..., t)
signal_bases(non::Tnon, tv::AbstractVector) =
    stack(t -> signal_bases(non, t), tv; dims=1);
# Signal model that combines nonlinear and linear effects:
signal(lin::Tlin, non::Tnon, tv::AbstractVector) =
    signal_bases(non, tv) * collect(lin);

# Signal model helpers:
signal(lin::AbstractVector, non::Tnon, tv::AbstractVector) =
   signal(Tlin(lin), non, tv)
signal(lin, non::AbstractVector, tv::AbstractVector) =
   signal(lin, Tnon(non), tv)
function signal(x::Tall, tv::AbstractVector)
   fun = name -> getfield(x, name)
#src @show fun.(Lin) fun.(Non)
   signal(Tlin(fun.(Lin)), Tnon(fun.(Non)), tv)
#src signal(collect(x), tv) # simpler, but riskier?
end
signal(x::AbstractVector, tv::AbstractVector) =
   signal(Tall(x), tv)
#src signal(x[1:length(Lin)], x[length(Lin)+1:end], tv) # ugly

# ## Simulate data:
y_true = signal(c_true, r_true, tm)
@assert y_true == signal(x_true, tm)
tf = Tf.(range(0, M, 201) * Δte) # fine sampling for plots
yf = signal(c_true, r_true, tf)
#src xaxis_t = ("t", (0,200).*ms, (0:4:M)*Δte) # units
xaxis_t = ("t [ms]", (0,200), (0:4:M)*Δte/ms) # no units
py = plot( xaxis = xaxis_t, yaxis = ("y", (0,100)) )
plot!(py, tf/ms, yf, color=:black)
scatter!(py, tm/ms, y_true, label = "Noiseless data, M=$M samples")

#=
## Random phase and noise
Actual MRI data has some phase and noise.
=#
phase_true = rand() * 2π + 0π
y_true_phased = Tc.(cis(phase_true) * y_true)

snr = 25 # dB
snr2sigma(db, y) = 10^(-db/20) * norm(y) / sqrt(length(y))
σ = Tf(snr2sigma(snr, y_true_phased))
yc = y_true_phased + σ * randn(Tc, M)
@show 20 * log10(norm(yc) / norm(yc - y_true_phased)) # check σ

# The phase of the noisy data becomes unreliable for low signal values:
pp = scatter(tm/ms, angle.(y_true_phased), label = "True data",
 xaxis = xaxis_t,
 yaxis = ("Phase", (-π, π), ((-1:1)*π, ["-π", "0", "π"])),
)
scatter!(tm, angle.(yc), label="Noisy data")


#
pc = scatter(tm/ms, real(yc),
 label = "Noisy data - real part",
 xaxis = xaxis_t,
 ylim = (-100, 100),
)
scatter!(pc, tm/ms, imag(yc),
 label = "Noisy data - imag part",
)



#=
## Phase correction
Phase correct signal using phase of first (noisy) data point
=#
yr = conj(sign(yc[1])) .* yc

pr = deepcopy(py)
scatter!(pr, tm/ms, real(yr),
 label = "Phase corrected data - real part",
 xaxis = xaxis_t,
 ylim = (-5, 105),
 marker = :square,
)
scatter!(pr, tm/ms, imag(yr),
 label = "Phase corrected data - imag part",
)


# Examine the distribution of real part after phase correction
function make1_phase_corrected_signal()
    phase_true = rand() * 2π
    y_true_phased = Tc.(cis(phase_true) * y_true)
    yc = y_true_phased + σ * randn(Tc, M)
    yr = conj(sign(yc[1])) .* yc
end

N = 2000
ysim = stack(_ -> make1_phase_corrected_signal(), 1:N)
tmp = ysim[end,:];

pe = scatter(real(tmp), imag(tmp), aspect_ratio=1,
 xaxis = ("real(y_$M)", (-4,8), -3:8),
 yaxis = ("imag(y_$M)", (-6,6), -5:5),
)
plot!(pe, real(y_true[end]) * [1,1], [-5, 5])
plot!(pe, [-5, 5] .+ 3, imag(y_true[end]) * [1,1])


# Histogram of the real part looks reasonably Gaussian
ph = histogram((real(tmp) .- real(y_true[end])) / (σ/√2), bins=-4:0.1:4,
 xlabel = "Real part of phase-corrected signal y_$M")

#src mean(real(tmp)) - y_true[end]

#src Log-linear fitting is inapplicable to bi-exponential models
#src Ts = u"s^-1"
#src Ts = Tf
#src roundr(rate) = round(Ts, Float64(Ts(rate)), digits=2)
roundr(rate) = round(rate, digits=2);
#src todo xaxis_td = ("Δt", (0,195).*ms, [0ms; tm_diff; M*Δte])


#=
## CRB
Compute CRB for precision of unbiased estimator.
This requires inverting the Fisher information matrix.
If the Fisher information matrix has units,
then Julia's built-in inverse `inv` does not work.
See 2.4.5.2 of Fessler 2024 book for tips.
=#

"""
Matrix inverse for matrix whose units are suitable for inversion,
meaning `X = Diag(left) * Z * Diag(right)`
where `Z` is unitless and `left` and `right` are vectors with units.
(Fisher information matrices always have this structure.)

[irrelevant when units are excluded]
"""
function inv_unitful(X::Matrix{<:Number})
    right = oneunit.(X[1,:]) # units for "right side" of matrix
    left = oneunit.(X[:,1] / right[1]) # units for "left side" of matrix
    N = size(X,1)
    Z = [X[i,j] / (left[i] * right[j]) for i in 1:N, j in 1:N]
    Zinv = inv(Z) # Z should be unitless if X has inverse-appropriate units
    Xinv = [Zinv[i,j] / (right[i] * left[j]) for i in 1:N, j in 1:N]
    return Xinv
end;


signal(x) = signal(x, tm)
@assert y_true == signal(collect(x_true))

# Jacobian of signal w.r.t. both linear and nonlinear parameters
jac = ForwardDiff.jacobian(signal, collect(x_true));

# Fisher information
fish = jac' * jac / σ^2;

# Compute CRB from Fisher information via matrix inverse
crb = inv_unitful(fish)

round3(x) = round(x; digits=3)
crb_std = Tall(round3.(sqrt.(diag(crb)))) # relabel CRB std. deviations


#=
## Dictionary matching

This approach is essentially a quantized maximum-likelihood estimator.
Here the quantization interval of 0.5/s
turns out to be much smaller
than the estimator standard deviation,
so the quantization error seems negligible.

Simple dot products
seem inapplicable
to a 2-pool model, so we use VarPro.

The
[VarPro](https://doi.org/10.1088/0266-5611/19/2/201)
cost function (for complex coefficients) becomes
```math
f(r) = -(A'y)' (A'A)^{-1} A'y
```
where ``A = A(r)`` is a ``M × 2`` matrix for each `r`.

By applying the QR decomposition of `A`,
the cost function simplifies to
``-‖Q'y‖₂``,
which is a natural extension of the dot product
used in dictionary matching.
=#
ra_list = Tf.(range(50/s, 160/s, 221)) # linear spacing?
rb_list = Tf.(range(0/s, 40/s, 81)) # linear spacing?
bases_unnormalized(ra, rb) = signal_bases((;ra, rb), tm)
dict = bases_unnormalized.(ra_list, rb_list');

# Plot the fast and slow dictionary components
tmp = stack(first ∘ eachcol, dict[:,1])
pd1 = plot(tm/ms, tmp[:,1:5:end]; xaxis=xaxis_t, marker=:o)
tmp = stack(last ∘ eachcol, dict[1,:])
pd2 = plot(tm/ms, tmp[:,1:5:end]; xaxis=xaxis_t, marker=:o)
pd12 = plot(pd1, pd2, plot_title = "Dictionary")

#
dict_q = map(A -> Matrix(qr(A).Q), dict)
dict_q = map(A -> sign(A[1]) * A, dict_q); # preserve sign of 1st basis

#
tmp = stack(first ∘ eachcol, dict_q[:,1])
pq1 = plot(tm/ms, tmp[:,1:5:end]; xaxis=xaxis_t, marker=:o)
tmp = stack(last ∘ eachcol, dict_q[1,:])
pq2 = plot(tm/ms, tmp[:,1:5:end]; xaxis=xaxis_t, marker=:o)
pq12 = plot(pq1, pq2, plot_title = "Orthogonalized Dictionary")

#
varpro_cost(Q::Matrix, y::AbstractVector) = -norm(Q'*y)
varpro_best(y) = findmin(Q -> varpro_cost(Q, y), dict_q)[2]

if !@isdefined(i_vp) # perform dictionary matching via VarPro
    i_vp = map(varpro_best, eachcol(ysim));
end
ra_dm = map(i -> ra_list[i[1]], i_vp) # dictionary matching estimates
rb_dm = map(i -> rb_list[i[2]], i_vp)

ph_ra_dm = histogram(ra_dm, bins=60:5:160,
 label = "Mean=$(roundr(mean(ra_dm))), σ=$(roundr(std(ra_dm)))",
 xaxis = ("Ra estimate via dictionary matching", (60, 160)./s),
)
plot!(r_true.ra*[1,1], [0, 3e2])
plot!(annotation = (140, 200, "CRB(Ra) = $(roundr(crb_std.ra))", :red))

#
ph_rb_dm = histogram(rb_dm, bins=14:0.5:26,
 label = "Mean=$(roundr(mean(rb_dm))), σ=$(roundr(std(rb_dm)))",
 xaxis = ("Rb estimate via dictionary matching", (14, 26)./s),
)
plot!(r_true.rb*[1,1], [0, 3e2])
plot!(annotation = (24, 200, "CRB = $(roundr(crb_std.rb))", :red))

#src gui(); throw(); # xx

#=
## Future work

- Compare to ML via VarPro
- Compare to ML via NLLS
- Cost contours, before and after eliminating x
- MM approach?
- GD?
- Newton's method?
- Units?
=#

include("../../inc/reproduce.jl")
