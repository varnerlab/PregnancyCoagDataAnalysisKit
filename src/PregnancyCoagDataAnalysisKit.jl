module PregnancyCoagDataAnalysisKit

    # include my codes -
    include("Include.jl")

    # export types -
    export VLResult

    # export functions -
    export load_study_data_set
    export extract_minimum_complete_data_set
    export z_score_transform_data_set
    export ols_fit_linear_model
    export mle_fit_logistic_model_classifier
    export mle_logistic_model_classifier_cross_validation
    export evaluate_classifier
    
end # module
