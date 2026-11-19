from agents.workflow import build_data_scientish_graph
from core.state import create_initial_state


def main():

    new_dataset_path = "storage/c.csv"  
    target_column_name = "Attrition"      


    
    app = build_data_scientish_graph()

    initial_state = create_initial_state(
        user_query=f"Execute end-to-end machine learning pipeline predicting '{target_column_name}'.",
        dataset_path=new_dataset_path
    )
    initial_state["cleaned_dataset_path"] = new_dataset_path
    initial_state["target_column"] = target_column_name

  
    final_state = app.invoke(initial_state)


if __name__ == "__main__":
    main()