# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
function _obj_function_logistics_regression(parameters::Array{Float64,1}, labels::Array{Int64,1}, 
    dataMatrix::Array{Float64,2}, bias::Float64 = 0.0)::Float64

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)
    prob_array = Array{Float64,1}(undef,number_of_rows)
    term_array = Array{Float64,2}(undef,number_of_rows,3)

    # augment the dataMatrix -
    X = [ones(number_of_rows) dataMatrix]
    
    # ok, so let's compute the log liklehood -
    # start w/the prob -
    for row_index = 1:number_of_rows
        prob_array[row_index] = _logistics_classifier_logic(parameters, X[row_index,:], bias)
    end
    
    # compute the term array -
    for row_index = 1:number_of_rows
        
        # compute term 1 and 2 -
        term_1_value = labels[row_index]*log(prob_array[row_index])
        term_2_value = (1 - labels[row_index])*log((1 - prob_array[row_index]))

        if (isnan(term_1_value) == true)
            term_1_value = 0.0
        end

        if (isnan(term_2_value) == true)
            term_2_value = 0.0
        end

        # package -
        term_array[row_index,1] = term_1_value
        term_array[row_index,2] = term_2_value
        term_array[row_index,3] = term_1_value + term_2_value
    end
    
    # check for inf -
    tmp = sum(term_array[:,3])
    if (isinf(tmp) == true)
        tmp = 0.0
    end

    # compute log liklehood -
    LL = -1*tmp

    # return -
    return LL
end

function _leave_one_out_logic(index::Int64,outputVector::Array{Float64,1}, dataMatrix::Array{Float64,2})::NamedTuple
end

function _evaluate_ols_linear_model(outputArray::Array{Float64,1}, dataMatrix::Array{Float64,2}, 
    paramaterArray::Array{Float64,1})::NamedTuple

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)
    Y_measured = outputArray

    # augement the data array 0
    ones_array = ones(number_of_rows)
    X = [ones_array dataMatrix]

    # compute the Y_model -
    Y_model = X*paramaterArray

    # what is the correlation?
    correlation_value = cor(Y_measured, Y_model)

    # what is the residual?
    residual_value = rmsd(Y_measured, Y_model)

    # package -
    results_tuple = (model_prediction=Y_model, residual=residual_value, correlation=correlation_value)

    # return -
    return results_tuple
end
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
"""
    mle_logistic_model_classifier_cross_validation(labelVector::Array{Int64,1}, dataMatrix::Array{Float64,2}, 
        numberOfGroups::Int64; selectionFunction::Union{Nothing, Function} = nothing)::VLResult
"""
function mle_logistic_model_classifier_cross_validation(labelVector::Array{Int64,1}, dataMatrix::Array{Float64,2}, 
    numberOfGroups::Int64; selectionFunction::Union{Nothing, Function} = nothing)::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try 

        ex = ErrorException("Ooops! mle_logistic_model_classifier_cross_validation not yet implemented.");
        throw(ex)

    catch error
        return VLResult(error)
    end
end

"""
    mle_fit_logistic_model_classifier(labelVector::Array{Int64,1}, dataMatrix::Array{Float64,2};
        initialParameterArray::Union{Nothing,Array{Float64,1}} = nothing, maxIterations::Int64=10000,
        showTrace::Bool = false, bias::Float64=0.0)::VLResult
"""
function mle_fit_logistic_model_classifier(labelVector::Array{Int64,1}, dataMatrix::Array{Float64,2};
    initialParameterArray::Union{Nothing,Array{Float64,1}} = nothing, maxIterations::Int64=10000,
    showTrace::Bool = false, bias::Float64=0.0)::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try 

        # setup the objective function -
        OF(p) = _obj_function_logistics_regression(p,labelVector,dataMatrix, bias)
        
        # setup initial guess -
        pinitial = 0.1*ones(number_of_cols+1)
        if (isnothing(initialParameterArray) == false)
            pinitial = initialParameterArray
        end

        # call the optimizer -
        opt_result = optimize(OF, pinitial, NelderMead(), 
            Optim.Options(iterations=maxIterations, show_trace=showTrace))

        # get the optimal parameters -
        β = Optim.minimizer(opt_result)
    
        # return -
        return VLResult(β)
    catch error
        return VLResult(error)
    end
end

"""
    ols_fit_linear_model(outputVector::Array{Float64,1}, dataMatrix::Array{Float64,2})::VLResult    
"""
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

        # compute some performance stuff -
        performance_tuple = _evaluate_ols_linear_model(outputVector,dataMatrix,theta)

        # setup results tuple -
        results_tuple = (model_prediction=performance_tuple.model_prediction, 
            correlation=performance_tuple.correlation, residual=performance_tuple.residual, 
            parameters=theta)

        # return -
        return VLResult(results_tuple)
    catch error
        return VLResult(error)
    end
end

"""
    ols_fit_linear_model_cross_validation(outputVector::Array{Float64,1}, 
        dataMatrix::Array{Float64,2}; numberOfGroups::Int64 = 0, selectionFunction::Union{Nothing, Function} = nothing)::VLResult
"""
function ols_fit_linear_model_cross_validation(outputVector::Array{Float64,1}, 
    dataMatrix::Array{Float64,2}; numberOfGroups::Int64 = 0, selectionFunction::Union{Nothing, Function} = nothing)::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)

    try

        # check: if numberOfGroups = 0, then set to the number of rows -
        if (numberOfGroups == 0)
            numberOfGroups = number_of_rows
        end

        # check: if selectionFunction == nothing, then use _leave_one_out_logic -
        if (isnothing(selectionFunction) == true)
            selectionFunction = _leave_one_out_logic
        end

        # initialize storage -
        parameter_storage_array = zeros(number_of_cols, numberOfGroups)
        total_residual_array = Array{Float64,1}()
        total_correlation_array = Array{Float64,1}()
        model_prediction_array = zeros(number_of_rows, numberOfGroups)
        
        # ok, lets go ...
        for group_index = 1:numberOfGroups
            
            # pass the full data set, and the group index to the selection function
            # the selectionFunction gives back the output and input arrsy -
            selection_tuple = selectionFunction(group_index,outputVector,dataMatrix)
            Yhat = selection_tuple.output_vector
            Xhat = selection_tuple.input_matrix

            # so let's scale these things -
            
            # YHat -
            scale_result = z_score_transform_vector(Yhat)
            if (isa(scale_result.value,Exception) == true)
                throw(scale_result.value)
            end
            Yhat_z_scaled = scale_result.value
            
            # Input array -
            scale_result = z_score_transform_array(Xhat)
            if (isa(scale_result.value,Exception) == true)
                throw(scale_result.value)
            end
            Xhat_z_scaled = scale_result.value

            # fit the model -
            fit_model_result = ols_fit_linear_model(Yhat_z_scaled, Xhat_z_scaled)
            if (isa(fit_model_result.value,Exception) == true)
                throw(fit_model_result.value)
            end
            performance_tuple = fit_model_result.value
            theta_parameters = performance_tuple.parameters
            model_output = performance_tuple.model_prediction

            # capture the model output -
            for (output_index, output_value) in enumerate(model_output)
                model_prediction_array[output_index, group_index] = output_value
            end

            # put the parameters in storage -
            for (parameter_index,parameter_value) in enumerate(theta_parameters)
                parameter_storage_array[parameter_index,group_index] = parameter_value
            end

            # grab the residual and other stuff for this group -
            push!(total_residual_array, performance_tuple.residual)
            push!(total_correlation_array, performance_tuple.correlation)
        end
        
        # return the results in a NamedTuple -
        results_tuple = (correlation=total_correlation_array,
            residual=total_residual_array,parameters=parameter_storage_array,
            model_prediction=model_prediction_array)

        # return -
        return VLResult(results_tuple)
    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #
