# setup activate -
# ...

# declare external packages -
using DataFrames
using CSV
using Statistics
using StatsBase
using LinearAlgebra
using Optim


# include my codes -
include(joinpath("base","Types.jl"))
include(joinpath("base","Checks.jl"))
include(joinpath("datastore","Files.jl"))
include(joinpath("analysis", "Pretreatment.jl"))
include(joinpath("analysis", "Regression.jl"))
include(joinpath("analysis", "Classify.jl"))