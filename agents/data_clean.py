from core.state import DataScientistState
from tools.datatool import clean_dataset_pipeline
def data_cleaning_node(state : DataScientistState):
    raw_dataset_path = state.get("raw_dataset_path")
    target_column = state.get("target_column" , "")
    output_path = "storage/cleaned_dataset.csv"
    clean_data = clean_dataset_pipeline(raw_dataset_path ,output_path, target_column)

    return{
        "cleaned_dataset_path" : output_path,
        "next_step" : "supervisor",
        "execution_logs" : [f"Data Cleaning Completed: Dropped {clean_data['duplicates_removed']} duplicates. "
            f"Preserved {clean_data['after_row']}/{clean_data['before_row']} rows. "
            f"Clean dataset saved to '{output_path}"]
    }