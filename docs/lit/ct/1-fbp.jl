#=
# [FBP Overview](@id 1-fbp)

This example illustrates how to perform filtered back-projection (FBP)
image reconstruction in CT
using the Julia language.

This is under construction.
See the demos in the Sinograms.jl package instead.
=#

#srcURL

#=
First we add the Julia packages that are need for these examples.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "ImagePhantoms"
        "Unitful"
        "Plots"
        "LaTeXStrings"
        "MIRTjim"
        "MIRT"
        "Sinograms"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

#=
using ImagePhantoms: shepp_logan, SheppLogan, radon, phantom
using Unitful: mm
using Plots; default(label="", markerstrokecolor=:auto)
using LaTeXStrings
using MIRT: diffl_map, ncg
using Sinograms: todo
=#
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


#
isinteractive() && prompt();


# Get the ellipse parameters for a CT-suitable version of the Shepp-Logan phantom
# and calculate (analytically) its sinogram.

#=
if false
object = shepp_logan(SheppLogan(); fovs=(FOV,FOV))
sino = radon(object).(r,ϕ')
data = data / oneunit(eltype(data)) # abandon units at this point
jim(kr, kϕ, abs.(data), title="k-space data magnitude",
    xlabel=L"k_r",
    ylabel=L"k_{\phi}",
    xticks = (-1:1) .* maximum(abs, kr),
    yticks = (0,π),
    ylims = (0,π),
    aspect_ratio = :none,
)
end
=#

include("../../inc/reproduce.jl")
