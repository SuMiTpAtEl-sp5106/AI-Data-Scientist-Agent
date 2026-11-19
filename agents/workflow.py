from core.state import DataScientistState
from agents.eda import eda_node
from agents.feature_engineering import feature_engineering_node
from agents.model_training import model_training_node
from agents.reporting import reporting_node

from langgraph.graph import StateGraph , START ,END
def supervisor_router(state: dict) -> str:
    if state.get("executive_report_path") or state.get("next_step") == "FINISH":
        return END
    elif state.get("best_model_path"):
        return "reporting_node"
    elif state.get("processed_data_path"):
        return "model_training_node"
    elif state.get("eda_insights"):
        return "feature_engineering_node"
    else:
        return "eda_node"


    

def build_data_scientish_graph():
    workflow = StateGraph(DataScientistState)

    workflow.add_node("eda_node" , eda_node)
    workflow.add_node("feature_engineering_node" , feature_engineering_node)
    workflow.add_node("model_training_node" , model_training_node)
    workflow.add_node("reporting_node" , reporting_node)

    workflow.set_entry_point("eda_node")

    workflow.add_conditional_edges(
        "eda_node",
        supervisor_router,
        {
            "feature_engineering_node" :"feature_engineering_node",
            "model_training_node" : "model_training_node",
            "reporting_node" : "reporting_node",
            END : END
        }
    )
    workflow.add_conditional_edges(
        "feature_engineering_node" ,
        supervisor_router,
        {
            "model_training_node" : "model_training_node",
            "reporting_node" : "reporting_node",
            END : END
        }
    )

    workflow.add_conditional_edges(
        "model_training_node",
        supervisor_router,

        {
        "reporting_node" : "reporting_node",
        END : END
        }
    )

    workflow.add_edge("reporting_node", END)

    return workflow.compile()




