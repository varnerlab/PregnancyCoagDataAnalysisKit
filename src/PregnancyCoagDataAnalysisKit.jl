module PregnancyCoagDataAnalysisKit

    # include my codes -
    include("Include.jl")

    # export types -
    export VLResult

    # export functions -
    export load_study_data_set
    export extract_minimum_complete_data_set
    export evaluate_classifier

    # data transform -
    export z_score_transform_data_set
    export z_score_transform_array
    export z_score_transform_vector

    # mle classifier -
    export mle_fit_logistic_model_classifier
    export mle_logistic_model_classifier_cross_validation

    # ols methods -
    export ols_fit_linear_model
    export ols_fit_linear_model_cross_validation
    
end # module
