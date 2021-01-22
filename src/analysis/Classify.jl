# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
function _logistics_classifier_logic(parameters::Array{Float64,1}, dataVector::Array{Float64,1})::Float64

    f = sum(dataVector.*parameters)
    T = exp(-f)
    prob_value = 1/(1+T)
    return prob_value
end
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function evaluate_classifier(parameters::Array{Float64,1}, dataMatrix::Array{Float64,2}; 
    classifierFunction::Union{Nothing,Function} = _logistics_classifier_logic)::VLResult

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
            predicted_classification = classifierFunction(parameters, dataVector)

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
    classifierFunction::Union{Nothing,Function} = _logistics_classifier_logic)::VLResult

    try 

        # augment the dataMatrix -
        X = [1.0 transpose(dataMatrix)]

        # compute the predicted classification -
        prob = classifierFunction(parameters, X)

        # return -
        return VLResult(prob)
    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #
