import pandas as pd
from langchain.tools import tool
import numpy as np

def inset_dataset(input_path : str):
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    num_row , num_col =df.shape
    numerical_features = df.select_dtypes(include="number").columns.tolist()
    categorical_features = df.select_dtypes(include="object").columns.tolist()

    missing_percentage =(( df.isnull().sum()/len(df))*100).round(2).to_dict()

    candidate_target = []

    taget = ["churn", "target", "label", "class", "status", "default"]

    for col in df.columns:
        if any (kw in col.lower() for kw in taget):
            candidate_target.append(col)

    if not candidate_target:
        candidate_target = [col for col in df.columns if 2 <= df[col].nunique() <= 10]

    return {
        "num_row": num_row,
        "num_col": num_col,
        "numerical_columns": numerical_features,
        "categorical_columns": categorical_features,
        "missing_value_pct": missing_percentage,
        "candidate_target": candidate_target
    }



def clean_dataset_pipeline(input_path : str , output_path : str , target_col : str  = None)-> dict:
    if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
    elif input_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_path)
    else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    before_row = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = before_row - len(df)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col]= df[col].str.strip().replace("" , np.nan)
        if col != target_col:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0 and converted.notna().sum() >= (df[col].notna().sum() - 10):
                df[col] = converted

    df = df.fillna(df.median(numeric_only = True))
    df =df.fillna({col : "Unknown" for col in df.select_dtypes(include="object").columns})

    after_row = df.shape[0]
    df.to_csv(output_path , index = False)

    return {
        "before_row" : before_row,
        "after_row" : after_row,
        "duplicates_removed" :  before_row - len(df),
        "output_path" : output_path
    }