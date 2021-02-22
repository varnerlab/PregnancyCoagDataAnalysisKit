# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
"""
    z_score_transform_data_set(dataFrame::DataFrame,colNameArray::Array{String,1})::VLResult
"""
function z_score_transform_data_set(dataFrame::DataFrame,colNameArray::Array{String,1})::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataFrame)
    z_score_array = Array{Float64,2}(undef,number_of_rows,length(colNameArray))

    try

        # ok:  for each col, lets z-center the data -
        for (col_index,col_name) in enumerate(colNameArray)
            
            # grab the data -
            data_col = dataFrame[!,col_name]

            # transform -
            tmp = parse.(Float64,data_col)
            
            # compute the mean, std -
            µ = mean(tmp)
            σ = std(tmp)

            # compute the z-score -
            z_score_col = (tmp .- μ)./σ

            # add -
            for (row_index,value) in enumerate(z_score_col)
                z_score_array[row_index,col_index] = value
            end
        end

        # return -
        return VLResult(z_score_array)
    catch error
        return VLResult(error)
    end
end

"""
    z_score_transform_array(dataArray::Array{Float64,2})::VLResult
"""
function z_score_transform_array(dataArray::Array{Float64,2})::VLResult

    # initialize -
    (number_of_rows, number_of_cols) = size(dataArray)
    z_score_array = Array{Float64,2}(undef, number_of_rows, number_of_cols)

    try

        for col_index = 1:number_of_cols
            
            # grab the data col -
            data_col = dataArray[:,col_index]

            # zscale this col -
            zscale_vector_result = z_score_transform_vector(data_col)
            if (isa(zscale_vector_result.value, Exception) == true)
                throw(zscale_vector_result.value)
            end
            scaled_col_data = zscale_vector_result.value

            # package, and go around again ...
            for (row_index, scaled_value) in enumerate(scaled_col_data)
                z_score_array[row_index,col_index] = scaled_value
            end
        end

        # return -
        return VLResult(z_score_array)
    catch error
        return VLResult(error)
    end
end

"""
    z_score_transform_vector(dataVector::Array{Float64,1})::VLResult
"""
function z_score_transform_vector(dataVector::Array{Float64,1})::VLResult

    # initalize -
    number_of_elements = length(dataVector)
    z_score_array = Array{Float64,1}(undef, number_of_elements)

    try 

        # compute the mean, std -
        µ = mean(dataVector)
        σ = std(dataVector)

        # scale -
        for index = 1:number_of_elements
            
            # get the raw value -
            raw_value = dataVector[index]
            
            # scale -
            z_score_array[index] =  (raw_value - µ)/σ
        end

        # rerurn -
        return VLResult(z_score_array)

    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #