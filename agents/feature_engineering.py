from tools.feature import engineer_feature_pipe_line
from core.state  import DataScientistState


def feature_engineering_node(state : DataScientistState):
    clean_data_path = state.get("cleaned_dataset_path")
    print(clean_data_path)
    target_col = state["target_column"]
    output_path = "storage/processed_data.pkl"
    result = engineer_feature_pipe_line(clean_path=clean_data_path , target_col= target_col ,output_path= output_path)
    
    return {
        "processed_data_path": output_path,
        "next_step": "supervisor",
        "execution_logs": [
            f"Feature Engineering Completed: Transformed {result['num_features']} features. "
            f"Train split: {result['train_shape']}, Test split: {result['test_shape']}. "
            f"Saved payload to '{output_path}'."
        ]
    }