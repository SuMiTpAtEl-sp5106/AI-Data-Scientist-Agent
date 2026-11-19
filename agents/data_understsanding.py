from core.state import DataScientistState,DatasetOverview
from tools.datatool import inset_dataset


def data_understanding_node(state : DataScientistState):
    raw_data_path = state.get("raw_dataset_path")

    info = inset_dataset( raw_data_path)
    dataset_meta = DatasetOverview(
    nums_row=info["num_row"],
    nums_col=info["num_col"],
    numerical_column=info["numerical_columns"],
    categorical_column=info["categorical_columns"],
    missing_value_pct=info["missing_value_pct"],
    candidate_target=info["candidate_target"],
    class_distribution=info.get("class_distribution", {})
    )
    target_col = state.get( "target_column")
    if not target_col and info["candidate_target"]:
        target_col = info["candidate_target"][0]

    return {
        "dataset_meta": dataset_meta,
        "target_column": target_col,
        "next_step": "supervisor",
        "execution_logs": [
            f"Data Understanding completed: {info['num_row']} rows, {info['num_col']} cols. Identified target: '{target_col}'"
        ]
    }