using PregnancyCoagDataAnalysisKit
using Test

# default test -
function default_test()::Bool
    return true
end


@testset "default_test_set" begin
    @test default_test() == true
end