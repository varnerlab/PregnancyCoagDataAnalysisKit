# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
function _logistics_classifier_logic(parameters::Array{Float64,1}, dataVector::Array{Float64,1}, bias::Float64)::Float64

    f = sum(dataVector.*parameters)
    T = exp(-(f-bias))
    prob_value = 1/(1+T)
    return prob_value
end
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function evaluate_classifier(parameters::Array{Float64,1}, dataMatrix::Array{Float64,2}; 
    classifierFunction::Union{Nothing,Function} = _logistics_classifier_logic, bias::Float64 = 0.0)::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataMatrix)
    probArray = Array{Float64,1}(undef,number_of_rows)

    try 

        ## augment the dataMatrix -
        X = [ones(number_of_rows) dataMatrix]

        # main loop -
        for row_index = 1:number_of_rows

            # get the dataVector -
            dataVector = X[row_index,:]

            # compute the predicted classification -
            predicted_classification = classifierFunction(parameters, dataVector, bias)

            # store -
            probArray[row_index] = predicted_classification
        end

        # return -
        return VLResult(probArray)
    catch error
        return VLResult(error)
    end
end

function evaluate_classifier(parameters::Array{Float64,1}, dataMatrix::Array{Float64,1}; 
    classifierFunction::Union{Nothing,Function} = _logistics_classifier_logic, bias::Float64 = 0.0)::VLResult

    # what is the length -
    number_of_cols = length(dataMatrix)
    tmp_array = Array{Float64,1}(undef,number_of_cols+1)

    try 
        
        # first element is 1 -
        tmp_array[1] = 1.0
        for (index,value) in enumerate(dataMatrix)
            tmp_array[index+1] = value
        end

        # compute the predicted classification -
        prob = classifierFunction(parameters, tmp_array, bias)

        # return -
        return VLResult(prob)
    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #
