execute = isempty(ARGS) || ARGS[1] == "run"

org, reps = :JuliaImageRecon, :Examples
eval(:(using $reps))
import Documenter
import Literate

# https://juliadocs.github.io/Documenter.jl/stable/man/syntax/#@example-block
ENV["GKSwstype"] = "100"
ENV["GKS_ENCODING"] = "utf-8"

# generate examples using Literate
lit = joinpath(@__DIR__, "lit")
src = joinpath(@__DIR__, "src")
gen = joinpath(@__DIR__, "src/generated")

base = "$org/$reps"
repo_root_url = "https://github.com/$base/blob/main"
nbviewer_root_url =
    "https://nbviewer.org/github/$base/tree/gh-pages/generated"
binder_root_url =
    "https://mybinder.org/v2/gh/$base/gh-pages?filepath=generated"

repo = eval(:($reps))
# preprocessing
inc1 = "include(\"../../inc/reproduce.jl\")"

function prep_markdown(str, root, file)
    inc_read(file) = read(joinpath("docs/inc", file), String)
    repro = inc_read("reproduce.jl")
    str = replace(str, inc1 => repro)
    urls = inc_read("urls.jl")
    file = joinpath(splitpath(root)[end], splitext(file)[1])
    tmp = splitpath(root)[end-2:end] # docs lit ??
    urls = replace(urls,
        "xxxrepo" => joinpath(repo_root_url, tmp...),
        "xxxnb" => joinpath(nbviewer_root_url, tmp[end]),
        "xxxbinder" => joinpath(binder_root_url, tmp[end]),
    )
    str = replace(str, "#srcURL" => urls)
end

function prep_notebook(str)
    str = replace(str, inc1 => "", "#srcURL" => "")
end

for (root, _, files) in walkdir(lit), file in files
    splitext(file)[2] == ".jl" || continue # process .jl files only
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit => gen))[1]
    Literate.markdown(ipath, opath;
        repo_root_url,
        preprocess = str -> prep_markdown(str, root, file),
        documenter = execute, # run examples
    )
    Literate.notebook(ipath, opath;
        preprocess = prep_notebook,
        execute = false, # no-run notebooks
    )
end


# Documentation structure
ismd(f) = splitext(f)[2] == ".md"
pages(folder) =
    [joinpath("generated/", folder, f) for f in readdir(joinpath(gen, folder)) if ismd(f)]

isci = get(ENV, "CI", nothing) == "true"

format = Documenter.HTML(;
    prettyurls = isci,
    edit_link = "main",
    canonical = "https://$org.github.io/$reps",
    assets = ["assets/custom.css"],
)

Documenter.makedocs(;
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
    Documenter.deploydocs(;
        repo = "github.com/$base",
        devbranch = "main",
        versions = nothing,
        forcepush = true,
        push_preview = true,
        # see https://$org.github.io/$repo.jl/previews/PR##
    )
else
    @warn "may need to: rm -r src/generated/"
end
