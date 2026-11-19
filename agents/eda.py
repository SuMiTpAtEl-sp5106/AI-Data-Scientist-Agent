from core.state import DataScientistState
from tools.eda_tool import generate_ede_report


def eda_node(state : DataScientistState):
    cleaned_dataset_path= state.get("cleaned_dataset_path")
    target_column = state.get("target_column")
    output_dir = "storage/plots"
    report = generate_ede_report(cleaned_dataset_path, target_column , output_dir)
    plot_count = len(report.get("saved_plots", []))

    return {"eda_insights":{
            "top_correlations": report.get("top_correlations"),
            "plots": report.get("saved_plots")
        },
        "next_step" : "supervisor",
        "execution_logs" : [
            f"EDA Completed: Generated {plot_count} plots in '{output_dir}'. "
            f"Top correlated feature with '{target_column}': {list(report.get('top_correlations', {}).keys())[:1]}"
        ]
        }