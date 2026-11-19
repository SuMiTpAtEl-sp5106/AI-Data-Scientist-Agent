from core.state import DataScientistState

from pydantic import BaseModel,Field 
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage , HumanMessage , ToolMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

class RouterOutput(BaseModel):
    next_agent: str = Field(
        ...,
        description="The exact name of the next agent node to execute, or 'FINISH'"
    )
    reasoning: str = Field(
        ...,
        description="Brief justification for selecting this agent based on current workflow state"
    )


def supervisor_node(state : DataScientistState):
    system_prompt = SystemMessage(
        content="""You are the Chief AI Data Scientist and Workflow Supervisor.

        Analyze:
        1. user_query: Understand the user's objective.
        2. Current state: Check data cleaning, EDA, feature engineering, model training, evaluation, critic approval, and completed outputs.
        3. Select the EXACT next agent based on execution dependencies.

        Allowed agents:
        - data_understanding
        - data_cleaning
        - eda
        - feature_engineering
        - ml_trainer
        - dl_trainer
        - evaluator
        - explainability
        - rag_researcher
        - critic
        - code_generator
        - reporter
        - FINISH

        Rules:
        - Never execute downstream tasks (like model training) before data cleaning unless explicitly instructed.
        - Send failed/rejected results back to the appropriate agent for rework.
        - Use 'critic' to validate ML model results before calling 'reporter'.
        - Call 'reporter' when analysis is complete to generate the final summary.
        - Return 'FINISH' only when all user objectives and report generations are done."""
            )

    human_prompt = HumanMessage(
        content=f"""
            User Query: {state.get('user_query')}
            messages: Annotated[List[BaseMessage], operator.add]
            execution_logs: Annotated[List[str], operator.add]
            Current Workflow State Overview:
            - Raw Dataset: {state.get('raw_dataset_path')}
            - Cleaned Dataset: {state.get('cleaned_dataset_path')}
            - Target Column: {state.get('target_column')}
            - Dataset Metadata Extracted: {state.get('dataset_meta') is not None}
            - Models Trained Count: {len(state.get('trained_models', []))}
            - Best Model ID: {state.get('best_model_id')}
            - Critic Reviews Count: {len(state.get('critic_reviews', []))}
            - Generated Plots Count: {len(state.get('generated_plots', []))}
            - Final Report Ready: {state.get('final_report_markdown') is not None}

            Execution Log Tail:
            {state.get('execution_logs', [])[-3:]}

            Select the exact next agent and provide your reasoning."""
                )
    response = llm.with_structured_output(RouterOutput).invoke([system_prompt  , human_prompt])
    return {"next_agent" : response.next_agent,
            "execution_logs": [f"Supervisor routed execution to: {response.next_agent}. Reason: {response.reasoning}"]
            }



