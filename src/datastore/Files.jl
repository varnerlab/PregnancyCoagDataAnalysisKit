
# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function load_study_data_set(filePath::String; 
    removeMissingColumn::Bool = true, missingValueSentinal::String = "#N/A")::VLResult

    # initialize -
    col_set_to_keep = Array{String,1}()

    try 
        
        # check: is filePath ok?
        check_result = is_file_path_ok(filePath)
        if (isnothing(check_result.value) == false)
            return check_result
        end
    
        # ok, if we get here we can load the file -
        df = CSV.read(filePath, DataFrame)

        # if removeMissingColumn = true, then lets go through each col of the data set.
        # if *all* the values in the col are missing, then remove that col -
        if (removeMissingColumn == false)
            return VLResult(df)
        end

        # go trhough the cols, are there #N/A
        (number_of_rows,number_of_cols) = size(df)
        for col_index = 1:number_of_cols

            # replace the missingValueSentinal w/missing -
            df = replace(df[!,col_index],missingValueSentinal=>missing)
        end

        
        # return -
        return VLResult(df)
    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #