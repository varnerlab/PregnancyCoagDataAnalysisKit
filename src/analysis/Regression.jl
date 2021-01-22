# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function ols_fit_linear_model(outputVector::Array{Float64,1}, dataMatrix::Array{Float64,2})::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try 

        # check: dimension mismatch -
        # ...

        # ok: we need to create the X matrix
        ones_array = ones(number_of_rows)
        X = [dataMatrix ones_array]

        # compute the parameters vector -
        XT = transpose(X)
        theta = inv(XT*X)*XT*outputVector

        # return -
        return VLResult(theta)

    catch error
        return VLResult(error)
    end

end
# =========================================================================================================== #
