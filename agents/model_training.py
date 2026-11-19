from core.state import DataScientistState
from tools.model_tool import train_and_evaluate_models

def model_training_node(state: DataScientistState) -> dict:
    """Agent node that trains models, evaluates performance, and saves the best model."""
    processed_path = state.get("processed_data_path") or "storage/processed_data.pkl"
    model_output_path = "storage/models/best_model.pkl"

    result = train_and_evaluate_models(
        processed_path=processed_path,
        model_output_path=model_output_path
    )

    best_name = result.get("best_model_name", "Unknown Model")
    best_score = result.get("best_model_score", result.get("best_score", 0.0))
    task_type = result.get("task_type", "Classification")

    return {
        "best_model_path": model_output_path,
        "best_model_name": best_name,
        "best_model_score": best_score,
        "task_type": task_type,
        "next_step": "supervisor",
        "execution_logs": [f"Trained champion model: {best_name} with score {best_score:.4f}"]
    }