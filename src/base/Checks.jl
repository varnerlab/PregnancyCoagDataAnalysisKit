function is_file_path_ok(path_to_file::String)::VLResult

    if (isfile(path_to_file) == false)

        # error message -
        error_message = "Ooops! $(path_to_file) does not exist."
        error_object = ArgumentError(error_message)

        # return -
        return VLResult{ArgumentError}(error_object)
    end

    # default: return nothing -
    return VLResult(nothing)
end