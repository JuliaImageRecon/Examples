#---------------------------------------------------------
# # [FBP Overview](@id 1-fbp)
#---------------------------------------------------------

#=
This example illustrates how to perform filtered back-projection (FBP)
image reconstruction in CT
using the Julia language.
=#

#=
This entire page was generated using a single Julia file:
[1-fbp.jl](@__REPO_ROOT_URL__/ct/1-fbp.jl).
=#
#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`1-fbp.ipynb`](@__NBVIEWER_ROOT_URL__/ct/1-fbp.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`1-fbp.ipynb`](@__BINDER_ROOT_URL__/ct/1-fbp.ipynb),

# This is under construction.

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


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
