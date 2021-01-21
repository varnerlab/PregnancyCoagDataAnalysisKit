
# === PRIVATE FUNCTIONS THAT ARE NOT EXPORTED =============================================================== #
# =========================================================================================================== #

# === PUBLIC FUNCTIONS THAT ARE EXPORTED ==================================================================== #
function extract_minimum_complete_data_set(dataFrame::DataFrame, colNameArray::Array{String,1})::VLResult

    # initialize -
    rows_to_keep_array = Array{Int64,1}()

    try 

        # ok: grab the cols of interest -
        df_tmp = dataFrame[!,colNameArray]

        # logic: we need to go through each row, if we econter *any* missing value, then we need stop and throwthat row out
        (number_of_rows, number_of_cols) = size(df_tmp)
        for row_index = 1:number_of_rows
            
            # tmp -
            tmp_array = Array{Any,1}()
            for col_index = 1:number_of_cols
                push!(tmp_array,df_tmp[row_index,col_index])
            end

            # ok, so let's walk through -
            bit_array = ismissing.(tmp_array)

            # do we have *any* 1's?
            idx_one = findall(x->x==1,bit_array)
            if (length(idx_one)==0)
                push!(rows_to_keep_array,row_index)            
            end
        end

        # ok: return filtered array -
        df_complete = dataFrame[rows_to_keep_array,colNameArray]

        # return -
        return VLResult(df_complete)
    catch error
        return VLResult(error)
    end

end


function load_study_data_set(filePath::String; 
    removeMissingColumn::Bool = true, missingValueSentinal::String = "#N/A")::VLResult

    # initialize -
    col_to_keep_array = Array{Int64,1}()
    row_to_keep_array = Array{Int64,1}()

    try 
        
        # check: is filePath ok?
        check_result = is_file_path_ok(filePath)
        if (isnothing(check_result.value) == false)
            return check_result
        end
    
        # ok, if we get here we can load the file -
        df = CSV.read(filePath, DataFrame)

        # what are the original names?
        col_name_array = names(df)

        # go trhough the cols, are there #N/A
        df_missing = DataFrame(replace(Matrix(df), "$(missingValueSentinal)"=>missing))

        # update the col headers, to set back the original names -
        rename!(df_missing, col_name_array)

        # if removeMissingColumn = true, then lets go through each col of the data set.
        # if *all* the values in the col are missing, then remove that col -
        if (removeMissingColumn == false)
            return VLResult(df_missing)
        end

        # process cols -
        (number_of_rows,number_of_cols) = size(df_missing)
        for col_index = 1:number_of_cols
            
            # grab the col -
            data_col = df_missing[!,col_index]

            # ok, so let's through -
            bit_array = ismissing.(data_col)
            idx_one = findall(x->x==1,bit_array)

            # if the length(idx_one) => the number of rows, then all rows are missing
            if (length(idx_one) != number_of_rows)
                push!(col_to_keep_array, col_index)
            end
        end

        # grab the no-missing cols -
        df_populated_cols = df_missing[!,col_to_keep_array]

        # ok, last - let's remove rows that are all missing -
        (number_of_rows,number_of_cols) = size(df_populated_cols)
        for row_index = 1:number_of_rows
            
            # grab the row -
            data_row = df_populated_cols[row_index,:]

            # check -
            for col_index = 1:number_of_cols
                data_row_value = data_row[col_index]
                if (ismissing(data_row_value) == false)
                    push!(row_to_keep_array, row_index)
                    break
                end
            end
        end

        # grab the non-missing rows -
        df_populated_row_and_cols = df_populated_cols[row_to_keep_array,:]

        # return -
        return VLResult(df_populated_row_and_cols)
    catch error
        return VLResult(error)
    end
end
# =========================================================================================================== #