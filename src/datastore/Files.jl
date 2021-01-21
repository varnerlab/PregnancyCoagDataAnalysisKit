
# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function load_study_data_set(filePath::String; removeMissingColumn::Bool = true)::VLResult

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

        # go trhough the cols, are there *non* missing elements?
        for col_name in names(df)
            
            # grab all the rows for this col -
            row_counter = 1
            should_continue_to_loop = true
            while (should_continue_to_loop == true)
                
                # grab col name if we have a *non* missing value -
                test_value = df[row_counter,col_name]
                if (ismissing(test_value) == false)
                    push!(col_set_to_keep,col_name)
                    should_continue_to_loop = false
                end
                
                # otherwise go around again -
                row_counter = row_counter + 1
            end
        end

        # ok, get only the non-missing cols -
        df_non_missing = df[:,col_set_to_keep]

        # return -
        return VLResult(df_non_missing)
    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #