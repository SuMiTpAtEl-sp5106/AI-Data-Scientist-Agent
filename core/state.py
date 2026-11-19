from typing import TypedDict , Optional , List , Annotated,Any ,Dict
from pydantic import BaseModel,Field 
import operator
from langchain_core.messages  import BaseMessage

class DatasetOverview(BaseModel):
    nums_row : int = Field (description="Total number of rows in the dataset")
    nums_col : int = Field(description= "total number of columns  in  the dataset")
    numerical_column : List[str] = Field(default_factory=list)
    categorical_column : List[str] = Field(default_factory=list)
    missing_value_pct: Dict[str, float]= Field(default_factory=dict, description="Column-wise missingness percentage")
    candidate_target : List[str] = Field(default_factory=list, description="Potential target variable columns")
    class_distribution: Optional[Dict[str, float]] = Field(default=None, description="Distribution of class labels if target is categorical")

class ModelArtifact(BaseModel):
    mode_id : str = Field(description="Unique model identifier, e.g., 'xgboost_v1'")
    model_type: str = Field(..., description="Algorithm name, e.g., 'XGBoost', 'RandomForest', 'ANN'")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics: accuracy, precision, recall, f1, roc_auc, rmse, etc.")
    artifact_path: str = Field(..., description="Saved file path on disk (e.g., 'storage/models/xgboost.joblib')")
    is_selected_for_prod: bool = Field(default=False)

class CriticFeedback(BaseModel):
    is_passed: bool = Field(..., description="True if validation checks pass, False if rework is required")
    issue_category: Optional[str] = Field(default=None, description="Category of issue detected: e.g., 'Data Leakage', 'Overfitting', 'Invalid Metric'")
    rationale: str = Field(..., description="Detailed critique explanation")
    actionable_recommendations: List[str] = Field(default_factory=list, description="Steps for downstream agent recovery")



class DataScientistState(TypedDict):
  
    user_query: str
    raw_dataset_path: str
    cleaned_dataset_path: Optional[str]
    target_column: Optional[str]
    next_step: str
    

    dataset_meta: Optional[DatasetOverview]
    trained_models: Annotated[List[ModelArtifact], operator.add]
    best_model_id: Optional[str]
    critic_reviews: Annotated[List[CriticFeedback], operator.add]
    

    generated_code_files: Dict[str, str]
    generated_plots: Annotated[List[str], operator.add]
    final_report_markdown: Optional[str]
    

    messages: Annotated[List[BaseMessage], operator.add]
    execution_logs: Annotated[List[str], operator.add]

def create_initial_state(user_query : str , dataset_path : str)->DataScientistState:
        return {
        "user_query": user_query,
        "raw_dataset_path":dataset_path,
        "cleaned_dataset_path":None,
        "target_column":None,
        "next_step": None,
        
    
        "dataset_meta":None,
        "trained_models": [],
        "best_model_id": None,
        "critic_reviews": [],
    
        "generated_code_files":{},
        "generated_plots":[],
        "final_report_markdown":None,
        
    
        "messages":[],
        "execution_logs": ["State initialized successfully."]
        }

