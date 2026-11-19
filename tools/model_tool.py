import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor , RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def train_and_evaluate_models(processed_path: str, model_output_path: str = "storage/models/best_model.pkl") -> Dict[str, Any]:
    

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run Day 7 first.")
        
    with open(processed_path, "rb") as f:
        data = pickle.load(f)


    X_train = pd.DataFrame(data["X_train"]).fillna(0)
    X_test = pd.DataFrame(data["X_test"]).fillna(0)
    y_train = pd.Series(data["y_train"]).values.ravel()
    y_test = pd.Series(data["y_test"]).values.ravel()
    unique_target_count = len(np.unique(y_train))
    is_classification = unique_target_count <= 10 and not pd.api.types.is_float_dtype(y_train)


    best_score = -float("inf")
    best_model_name = ""
    best_model = None

    if is_classification:
        print(f"[*] Task Detected: Classification")
        models_to_test = {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=1000, random_state=42),
                "params": {"C": [0.1, 1.0, 10.0]}
            },
            "RandomForestClassifier": {
                "model": RandomForestClassifier(random_state=42),
                "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
            }
        }
        scoring_metric = "accuracy"
    else:
        print(f"[*] Task Detected: Regression")
        models_to_test = {
            "Ridge": {
                "model": Ridge(),
                "params": {"alpha": [0.1, 1.0, 10.0]}
            },
            "RandomForestRegressor": {
                "model": RandomForestRegressor(random_state=42),
                "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
            }
        }
        # CRITICAL: Use 'r2' for regression to prevent probability attribute errors!
        scoring_metric = "r2"

    # 2. Train and tune with the correct scoring metric
    for name, config in models_to_test.items():
        print(f"[*] Training and tuning {name} using scoring='{scoring_metric}'...")
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring=scoring_metric)
        grid.fit(X_train, y_train)
        
        score = grid.score(X_test, y_test)
        print(f"[✓] {name} Test Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = grid.best_estimator_
    if best_model is None:
            raise RuntimeError("All candidate models failed to train or evaluate successfully.")
    
        # 4. Save Champion Model
    model_output_path = "storage/best_model.pkl"
    os.makedirs("storage", exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"[✓] Saved champion model to {model_output_path}")


    return {
        "best_model_name": best_model_name,
        "best_model_score": float(best_score),
        "best_model_path": model_output_path,
        "task_type": "Classification" if is_classification else "Regression",
    
        "best_score": float(best_score) 
    }