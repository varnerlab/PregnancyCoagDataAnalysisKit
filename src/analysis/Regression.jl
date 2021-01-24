# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
function _obj_function_logistics_regression(parameters::Array{Float64,1}, labels::Array{Int64,1}, 
    dataMatrix::Array{Float64,2})::Float64

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)
    prob_array = Array{Float64,1}(undef,number_of_rows)
    term_array = Array{Float64,2}(undef,number_of_rows,3)

    # augment the dataMatrix -
    X = [ones(number_of_rows) dataMatrix]
    
    # ok, so let's compute the log liklehood -
    # start w/the prob -
    for row_index = 1:number_of_rows
        
        f = sum(X[row_index,:].*parameters)
        T = exp(-f)
        prob_array[row_index] = 1/(1+T)
    end
    
    # compute the term array -
    for row_index = 1:number_of_rows
        
        # compute term 1 and 2 -
        term_1_value = (prob_array[row_index])^(labels[row_index])
        term_2_value = (1 - prob_array[row_index])^(1 - labels[row_index])

        # package -
        term_array[row_index,1] = term_1_value
        term_array[row_index,2] = term_2_value
        term_array[row_index,3] = (term_1_value)*(term_2_value)
    end
    
    # compute log liklehood -
    LL = -log(prod(term_array[:,3]))

    # return -
    return LL
end

function _leave_one_out_logic()
end
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function mle_logistic_model_classifier_cross_validation(labelVector::Array{Int64,1}, dataMatrix::Array{Float64,2}, 
    numberOfGroups::Int64; selectionFunction::Union{Nothing, Function} = _leave_one_out_logic)::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try 

        ex = ErrorException("Ooops! mle_logistic_model_classifier_cross_validation not yet implemented.");
        throw(ex)

    catch error
        return VLResult(error)
    end
end

function mle_fit_logistic_model_classifier(labelVector::Array{Int64,1}, dataMatrix::Array{Float64,2};
    initialParameterArray::Union{Nothing,Array{Float64,1}} = nothing)::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try 

        # setup the objective function -
        OF(p) = _obj_function_logistics_regression(p,labelVector,dataMatrix)
        
        # setup initial guess -
        pinitial = 1.0*ones(number_of_cols+1)
        if (isnothing(initialParameterArray) == false)
            pinitial = initialParameterArray
        end

        # call the optimizer -
        opt_result = optimize(OF, pinitial, NelderMead())

        # get the optimal parameters -
        β = Optim.minimizer(opt_result)
    
        # return -
        return VLResult(β)
    catch error
        return VLResult(error)
    end
end

function ols_fit_linear_model(outputVector::Array{Float64,1}, dataMatrix::Array{Float64,2})::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try 

        # check: dimension mismatch -
        # ...

        # ok: we need to create the X matrix
        ones_array = ones(number_of_rows)
        X = [ones_array dataMatrix]

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
