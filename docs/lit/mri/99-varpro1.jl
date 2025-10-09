#=
# [Variable Projection: One exponential](@id varpro1)

Illustrate fitting one exponential to data.

See:
- [VarPro blog](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals)
- [VP4Optim.jl](https://github.com/cganter/VP4Optim.jl) has biexponential fits
- [Varpro.jl](https://github.com/macd/Varpro.jl)

=#

#srcURL

# ### Setup

# Packages needed here.

using ImagePhantoms: ellipse, ellipse_parameters, phantom, SheppLoganBrainWeb
using Statistics: mean, std
using Plots: default, gui, histogram, plot, plot!, scatter, scatter!
using Plots: cgrad, RGB
default(markerstrokecolor=:auto, label="", widen = true)
using LaTeXStrings
using LinearAlgebra: norm, Diagonal, diag, diagm
using MIRTjim: jim
using Random: seed!; seed!(0)
using Unitful: @u_str, uconvert, ustrip, ms, s, mm
using InteractiveUtils: versioninfo


#=
## Single exponential

We explore a simple case:
fitting a single exponential to some noisy data:
``y_m = x e^{- r t_m} + ϵ_m``
for ``m = 1,…,M``.
The two unknown parameters here are:
- the decay rate ``r ≥ 0``
- the amplitude ``x`` (that could be complex in some MRI settings)
=#

Tf = Float32
Tc = Complex{Tf}
M = 8 # how many samples
Δte = 25ms # echo spacing
te1 = 5ms # time of first echo
tm = Tf.(te1 .+ (0:(M-1)) * Δte) # echo times
x_true = 100 # AU
r_true = 20/s
signal(x, r; t=tm) = x * exp.(-t * r) # signal model
y_true = signal(x_true, r_true)
tf = range(0, M, 201) * Δte
xaxis_t = ("t", (0,200).*ms, [0ms; tm; M*Δte])
py = plot( xaxis = xaxis_t )
plot!(py, tf, signal(x_true, r_true; t=tf), color=:black)
scatter!(py, tm, y_true, label = "Noiseless data, M=$M samples")


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

pp = scatter(tm, angle.(y_true_phased), label = "True data",
 xaxis = xaxis_t,
 yaxis = ("Phase", (-π, π), ((-1:1)*π, ["-π", "0", "π"])),
)
scatter!(tm, angle.(yc), label="Noisy data")


# The phase of the noisy data becomes unreliable for low signal values
pc = scatter(tm, real(yc),
 label = "Noisy data - real part",
 xaxis = xaxis_t,
 ylim = (-100, 100),
)
scatter!(pc, tm, imag(yc),
 label = "Noisy data - imag part",
)


#=
## Phase correction
Phase correct signal using phase of first (noisy) data point
=#
yr = conj(sign(yc[1])) .* yc

pr = deepcopy(py)
scatter!(pr, tm, real(yr),
 label = "Phase corrected data - real part",
 xaxis = xaxis_t,
 ylim = (-5, 105),
 marker = :square,
)
scatter!(pr, tm, imag(yr),
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
tmp = ysim[end,:]

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


#=
## Log-linear fit
LS fitting using log of absolute value of real part
of phase-corrected data.
=#

yl = @. log(abs(yr))
pl = plot(xaxis=xaxis_t, yaxis=("Log data"))
log_fine = log.(signal(x_true, r_true; t=tf))
plot!(pl, tf, log_fine, color=:lightgray)
scatter!(pl, tm, log.(y_true), label="True", color=:black)
scatter!(pl, tm, yl, label="Noisy", color=:red)

#=
Linear (not affine!) fit,
after normalizing data by 1st data point
(which is a bit smaller than `x_true`
since first sample is not at ``t=0``).
=#
tm_diff = tm .- te1 # time shift by 1st echo
A1 = reshape(-tm_diff, M, 1)
yl0 = yl .- yl[1] # normalize
A1pinv = inv(A1'*A1) * A1'
r1 = A1pinv * yl0
r1 = only(r1)

Ts = u"s^-1"
xaxis_td = ("Δt", (0,195).*ms, [0ms; tm_diff; M*Δte])
pf1 = plot(xaxis=xaxis_td, yaxis=("Log data"))
plot!(pf1, tf .- te1, log_fine .- log(y_true[1]), color=:lightgray)
scatter!(pf1, tm_diff, log.(y_true/y_true[1]),
 label="True for R1=$r_true", color=:black)
scatter!(pf1, tm_diff, yl0, label = "Noisy", color=:red)
roundr(rate) = round(Ts, Float64(Ts(rate)), digits=2)
plot!(pf1, tf .- te1, -r1 .* (tf .- te1),
 label = "Linear Fit: R1 = $(roundr(r1))")


#=
Maybe that poor fit was just one unlucky trial?
Repeat the log-linear fit many times.
=#
ysim_log = log.(abs.(ysim))
ysim_log .-= ysim_log[1,:]'
r1sim = vec(A1pinv * ysim_log)
r1sim = Ts.(r1sim)

ph1 = histogram(r1sim, bins=16:0.2:32,
 label = "Mean=$(roundr(mean(r1sim))), σ=$(roundr(std(r1sim)))",
 xaxis = ("R1 estimate via log-linear fit", (16, 32)./s, (16:2:32)./s),
)
plot!(r_true*[1,1], [0, 140])


#=
## CRB
Compute CRB for precision of unbiased estimator.
This requires inverting the Fisher information matrix.
Here the Fisher information matrix has units,
so Julia's built-in inverse `inv` does not work.
See 2.4.5.2 of Fessler 2024 book for tips.
=#

"""
Matrix inverse for matrix whose units are suitable for inversion,
meaning `X = Diag(left) * Z * Diag(right)`
where `Z` is unitless and `left` and `right` are vectors with units.
(Fisher information matrices always have this structure.)
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

# Gradients of `y_true = x_true * exp.(- r_true * tm)`
grad1 = @. exp(-r_true * tm) # x
grad2 = @. -x_true * tm * exp(-r_true * tm) # r
grad = [grad1 grad2]
fish1 = grad' * grad / σ^2

# Compute CRB from Fisher information via unitful matrix inverse
crb = inv_unitful(fish1)
crb_r1 = Ts(sqrt(crb[2,2]))
crb_r1_xknown = Ts(sqrt(1/fish1[2,2])) # CRB if M0 is known, about 0.72

plot!(annotation = (27, 100,
 "CRB = $(roundr(crb_r1)) or $(roundr(crb_r1_xknown))", :red))



#=
## Log-linear fit to early echoes
Try discarding some of the later echoes
that have worse SNR.
The results depend a lot on how many echoes are used.
In this case, K = 4 or 5 works well,
but how would you know in practice?
=#

K = 3 # keep first few echos
A1K = A1[1:K,:]
A1pinvK = inv(A1K'*A1K) * A1K'
r1simK = vec(A1pinvK * ysim_log[1:K,:])
r1simK = Ts.(r1simK)

phK = histogram(r1simK, bins=16:0.2:32,
 label = "Mean=$(roundr(mean(r1simK))), σ=$(roundr(std(r1simK)))",
 xaxis = ("R1 estimate via log-linear fit K=$K", (16, 32)./s, (16:2:32)./s),
)
plot!(r_true*[1,1], [0, 140])
plot!(annotation = (27, 100,
 "CRB = $(roundr(crb_r1)) or $(roundr(crb_r1_xknown))", :red))


#=
## Compare to dictionary matching
This approach is essentially a quantized maximum-likelihood estimator.
Here the quantization interval of 0.1/s
turns out to be much smaller
than the estimator standard deviation of 0.6/s,
so the quantization error seems negligible.

It is essential to normalize the dictionary atoms
by the Euclidean norm.
The reason why is exactly the same as the
[VarPro](https://doi.org/10.1088/0266-5611/19/2/201)
derivation.
=#
r_list = range(0/s, 40/s, 401) # linear spacing?
dict = signal(1, r_list')
dict_norm = dict ./ norm.(eachcol(dict))'
pd = plot(
  plot(tm, dict[:,1:10:end], title="Dictionary"),
  plot(tm, dict_norm[:,1:10:end], title="Normalized Dictionary"),
)

# Inner products with (normalized) dictionary atoms
tmp = dict_norm' * real(ysim)
tmp = argmax.(eachcol(tmp)) # find max
r_dm = r_list[tmp]; # R2 value from dictionary

ph_dm = histogram(r_dm, bins=16:0.2:32,
 label = "Mean=$(roundr(mean(r_dm))), σ=$(roundr(std(r_dm)))",
 xaxis = ("R1 estimate via dictionary matching", (16, 32)./s, (16:2:32)./s),
)
plot!(r_true*[1,1], [0, 140])
plot!(annotation = (27, 100,
 "CRB = $(roundr(crb_r1)) or $(roundr(crb_r1_xknown))", :red))

#=
## WLS for log-linear fit
Compare to WLS via error propagation
Much faster than dictionary matching!

Derive weights.
The log of the normalized signal value
(ignoring phase correction and the absolute value)
is approximately
(using a 1st-order Taylor expansion)
```math
v_m
= \log(y_m / y_1)
= \log(y_m) - \log(y_1)
≈ \log(\bar{y}_m) - \log(\bar{y}_1)
+ (1/\bar{y}_m) (y_m - \bar{y}_m)
- (1/\bar{y}_1) (y_1 - \bar{y}_1)
```
so the variance of that log value is
```math
\mathrm{Var}(v_m)
≈ (1/\bar{y}_m^2) \, \mathrm{Var}(y_m - \bar{y}_m)
+ (1/\bar{y}_1^2) \, \mathrm{Var}(y_1 - \bar{y}_1)
= σ^2 \, ( 1/\bar{y}_m^2 + 1/\bar{y}_1^2 ) .
```
We would like to perform the WLS fit
using the reciprocal of that variance.
However,
the expected values
``\bar{y}_m = \mathbb{E}(y_m)`` are unknown
because they depend
on the latent parameter(s).

In practice we use a "plug-in" estimate
using the observed data
in place of the expectation:
```math
w_m = 1 / ( 1/y_m^2 + 1/y_1^2 ) .
```
The noise variance ``σ^2`` is the same for all ``m``
so it is irrelevant to the WLS fit:
```math
\arg\min_{x} (A x - b)' W (A x - b)
```
where here ``b`` is the log normalized data
``b_m = \log(y_m / y_1)``
and ``x`` here denotes the rate parameter ``r``.

For ``m=1`` we have ``v_1 = 1`` by construction,
which has zero variance.
Our log-linear model
``-r Δt``
is explicitly ``0`` at ``Δt_1 = 0``,
i.e., ``A_{1,1} = 0``,
so that first (normalized) data point provides no information
and is inherently excluded
from the linear fit
(for any weighting).
=#

"""
   wls_exp_fit(y)
Fit rate of a single exponential using weighted least-squares (WLS)
Expects non-log data.
Uses global `A1`.
Returns scalar rate estimate.
"""
function wls_exp_fit(y)
    w = @. (1 / y^2 + 1 / y[1]^2) # drop irrelevant σ^2
    w = 1 ./ w # weights via 1st-order Taylor approx
    A1w = w .* A1
    ylog = @. log(abs(y ./ y[1]))
    return only(A1w' * ylog) / only(A1w' * A1)
end

r1_wls = Ts.(wls_exp_fit.(real(eachcol(ysim))))

ph_wls = histogram(r1_wls, bins=16:0.2:32,
 label = "Mean=$(roundr(mean(r1_wls))), σ=$(roundr(std(r1_wls)))",
 xaxis = ("R1 estimate via log-WLS", (16, 32)./s, (16:2:32)./s),
)
plot!(r_true*[1,1], [0, 270])
plot!(annotation = (27, 100,
 "CRB = $(roundr(crb_r1)) or $(roundr(crb_r1_xknown))", :red))


#=
## Phantom illustration
=#

r2_values = [20, 0, -3, -4, 5, 6, 7, 8, 9, -2] / s
fovs = (256mm, 256mm)
params = ellipse_parameters(SheppLoganBrainWeb(); fovs, disjoint=true)
params = [(p[1:5]..., r2_values[i]) for (i, p) in enumerate(params)]
ob = ellipse(params)
y = range(-fovs[2]/2, fovs[2]/2, 256)
x = range(-205/2, 205/2, 206) * (y[2]-y[1])
oversample = 3
r2_map_true = phantom(x, y, ob[1:1], 1) + # trick to avoid edge issues
    phantom(x, y, ob[3:end], oversample)
mask = r2_map_true .> 0/s
x_map_true = @. 100 * mask * cis(π/2 + x / 100mm) # non-uniform phase

climr = (0,30) ./ s
p2r = jim(x, y, r2_map_true, "R2* Map";
 clim = climr, xlabel="x", ylabel="y", color=:cividis)
p2 = plot(
 p2r,
 jim(x, y, x_map_true, "|M0 Map|"; xlabel="x", ylabel="y"),
 jim(x, y, angle.(x_map_true), "∠M0 Map"; xlabel="x", ylabel="y", color=:hsv);
 layout = (1,3), size=(800,300),
)


# Simulate multi-echo data
y_true_phantom = signal.(x_map_true, r2_map_true)
y_true_phantom = stack(y_true_phantom)
y_true_phantom = permutedims(y_true_phantom, [2, 3, 1]) # (nx,ny,M)
dim = size(y_true_phantom)
y_phantom = y_true_phantom + σ * randn(Tc, dim...)

pyp = plot(
 jim(x, y, y_phantom; title="|Echo images|"),
 jim(x, y, angle.(y_phantom); title="∠Echo images", color=:hsv),
 layout = (1,2), size=(800,300),
)

# Phase correct
y_phantom_dephased = @. conj(sign(y_phantom[:,:,1])) * y_phantom
jim(x, y, angle.(y_phantom_dephased); title="∠Echo images after dephasing", color=:hsv)

# Take real part
y_phantom_realed = real(y_phantom_dephased)
jim(x, y, y_phantom_realed; title="Real(Echo images) after dephasing")

# WLS estimate of R2* from real part
r2_map_wls = Ts.(wls_exp_fit.(eachslice(y_phantom_realed; dims=(1,2))))
r2_map_wls .*= mask
RGB255(args...) = RGB((args ./ 255)...)
ecolor = cgrad([RGB255(230, 80, 65), :black, RGB255(23, 120, 232)])
rmse = sqrt(mean(abs2.(r2_map_wls[mask] - r2_map_true[mask])))
plot(p2r,
 jim(x, y, r2_map_wls, "R2* Map via WLS";
  clim = climr, xlabel="x", ylabel="y", color=:cividis),
 jim(x, y, r2_map_wls - r2_map_true, "R2* Error \n RMSE=$(roundr(rmse))";
  clim = (-3, 3) ./ s, color=ecolor);
 layout = (1,3), size=(800,300),
)


#=
## Future work

- affine fit via LS and WLS and ML

Biexponential case:
- Compare to ML via VarPro
- Compare to ML via NLS
- Cost contours, before and after eliminating x
- MM approach?
- GD?
- Newton's method?
=#

include("../../../inc/reproduce.jl")
