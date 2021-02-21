using Documenter, PregnancyCoagDataAnalysisKit

makedocs(sitename="Documentation")

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/varnerlab/PregnancyCoagDataAnalysisKit.jl.git",
    devbranch = "main",
    devurl = "dev",
)