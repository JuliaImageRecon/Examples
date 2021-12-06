using Documenter
using Literate

# https://juliadocs.github.io/Documenter.jl/stable/man/syntax/#@example-block
ENV["GKSwstype"] = "100"
ENV["GKS_ENCODING"] = "utf-8"

# generate examples using Literate
lit = joinpath(@__DIR__, "lit")
src = joinpath(@__DIR__, "src")
gen = joinpath(@__DIR__, "src/generated")

for (root, _, files) in walkdir(lit), file in files
    splitext(file)[2] == ".jl" || continue # process .jl files only
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit => gen))[1]
    Literate.markdown(ipath, opath, documenter = execute) # run examples
    Literate.notebook(ipath, opath; execute = false) # no-run notebooks
end

# Documentation structure
ismd(f) = splitext(f)[2] == ".md"
pages(folder) =
    [joinpath("generated/", folder, f) for f in readdir(joinpath(gen, folder)) if ismd(f)]

isci = get(ENV, "CI", nothing) == "true"

format = Documenter.HTML(;
    prettyurls = isci,
    edit_link = "main",
    canonical = "https://juliaimagerecon.github.io/Examples",
#   assets = String[],
)

makedocs(;
    modules = Module[],
    authors = "Jeff Fessler and contributors",
    sitename = "Examples",
    format,
    pages = [
        "Home" => "index.md",
        "MRI" => pages("mri"),
        "CT" => pages("ct"),
    ],
)

if isci
    deploydocs(;
        repo = "github.com/JuliaImageRecon/Examples.git",
        devbranch = "main",
        versions = nothing,
        forcepush = true,
        push_preview = true,
    )
else
    @warn "may need to: rm -r src/generate/"
end
