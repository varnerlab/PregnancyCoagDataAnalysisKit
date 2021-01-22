# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #

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
# =========================================================================================================== #