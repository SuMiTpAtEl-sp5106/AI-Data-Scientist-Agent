from core.state import DataScientistState
import pandas as pd 
import os 
import numpy as np
import pickle
from itertools import combinations
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def engineer_feature_pipe_line(clean_path: str, target_col: str, output_path: str = "storage/processed_data.pkl") -> dict:
  
    if clean_path.endswith('.csv'):
        df = pd.read_csv(clean_path)
    elif clean_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(clean_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not target_col or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset! Available columns: {list(df.columns)}")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        unique_vals = df[target_col].dropna().unique()
        if len(unique_vals) == 2:
            pos_val = unique_vals[0] if any(keyword in str(unique_vals[0]).lower() for keyword in ['yes', 'true', '1', 'positive', 'churn', 'survived']) else unique_vals[1]
            df[target_col] = (df[target_col] == pos_val).astype(int)
        else:
            
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col].astype(str))
    else:
        unique_count = df[target_col].nunique(dropna=True)
        if unique_count <= 10 and not pd.api.types.is_float_dtype(df[target_col]):
            df[target_col] = df[target_col].fillna(0).astype(int)
        else:
            df[target_col] = df[target_col].fillna(df[target_col].mean())

    
   
    id_cols = [col for col in df.columns if col.lower() == 'id' or col.lower().endswith('_id') or col.lower().startswith('id_')]
    df = df.drop(columns=id_cols, errors='ignore')

    #  Separate Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if X.empty or X.shape[1] == 0:
        raise ValueError("Feature matrix X is empty after preprocessing! Check your columns.")
    
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
   
    if len(num_cols) > 10:
        num_cols = num_cols[:10]
        
    ratio_dict = {}
    for col1, col2 in combinations(num_cols, 2):
        col_name = f"{col1}_ratio_{col2}"
        ratio_dict[col_name] = (X[col1] / X[col2].replace(0, np.nan)).fillna(0)
    
    if ratio_dict:
        ratio_df = pd.DataFrame(ratio_dict, index=X.index)
        X = pd.concat([X, ratio_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

   
    processed_payload = {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": X.columns.tolist()
    }

    with open(output_path, "wb") as f:
        pickle.dump(processed_payload, f)

    print(f"[✓] Log: Feature Engineering Completed. Total Features: {X.shape[1]}. Train: {X_train_df.shape}, Test: {X_test_df.shape}")

    return {
        "num_features": X.shape[1],
        "train_shape": X_train_df.shape,
        "test_shape": X_test_df.shape,
        "output_path": output_path
    }