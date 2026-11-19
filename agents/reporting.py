from core.state import DataScientistState
from tools.interpretation import generate_executive_report

def reporting_node(state : DataScientistState)-> dict:
    model_path = state.get("best_model_path") or "storage/models/best_model.pkl"
    processed_path = state.get("processed_data_path") or "storage/processed_data.pkl"
    output_path = "storage/reports/executive_summary.md"

    result = generate_executive_report(
        model_path=model_path,
        processed_path=processed_path,
        output_path=output_path
    )
    
    return {
        "executive_report_path": output_path,
        "next_step": "FINISH",
        "execution_logs": [
            f"Executive Reporting Completed: Compiled findings for {result['features_analyzed']} features. "
            f"Report saved to '{output_path}'."
        ]
    }