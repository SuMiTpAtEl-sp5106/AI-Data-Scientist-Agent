import pandas as pd 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

def generate_ede_report(dataset_path: str, target_col: str, output_dir: str = "storage/plots") -> dict:
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    num_df = df.select_dtypes(include=["number"])
    stats_dict = {}
    if not num_df.empty:
        stats = num_df.describe().T
        stats["skewness"]  = num_df.skew()
        stats_dict = stats[['mean', 'std', 'min', 'max', 'skewness']].to_dict("index")

    cat_df = df.select_dtypes(include=["object","string" , "category"])
    cat_dict = {}
    if not  cat_df.empty:
        cat_stats = cat_df.describe().T

        cat_dict = cat_stats[["count" , "unique"]].to_dict("index")

    top_corr= {}
    if target_col in df.columns:
        df_corr = num_df.copy()
        if df[target_col].dtype == 'object':
            df_corr[target_col] = (df[target_col].astype(str).str.lower() == 'yes').astype(int)
        
        if target_col in df_corr.columns:
            top_corr = (
                df_corr.corr()[target_col]
                .drop(target_col, errors="ignore")
                .sort_values(key=abs, ascending=False)
                .head(10)
            )
            top_corr_dict = top_corr.to_dict()


    if target_col in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x=target_col, hue=target_col, palette="viridis", legend=False)
        plt.title(f"Target Distribution: {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        target_plot_path = os.path.join(output_dir , "target_distribution.png")
        plt.savefig(target_plot_path)
        plt.close()
        saved_plots.append(target_plot_path)

    if len(num_df.columns)>=2:
        plt.figure(figsize=(8,6))
        corr_matrix = num_df.corr()
        sns.heatmap(corr_matrix , annot= True , fmt = ".2f" , cmap = "coolwarm" , cbar = True)
        plt.title("Numerical Feature Correlation Heatmap")
        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        saved_plots.append(heatmap_path)

    num_cols = [c for c in num_df.columns if c != target_col]
    if target_col in df.columns and len(num_cols) > 0:
        top_feature = num_cols[0]
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x=target_col, y=top_feature, hue=target_col, palette="Set2", legend=False)
        plt.title(f"Feature Interaction: {top_feature} vs {target_col}")
        plt.tight_layout()
        
        interaction_plot_path = os.path.join(output_dir, "feature_interaction.png")
        plt.savefig(interaction_plot_path)
        plt.close()
        saved_plots.append(interaction_plot_path)

    return {
        "numeric_stats": stats_dict,
        "categorical_stats": cat_dict,
        "top_correlations": top_corr,
        "saved_plots": saved_plots
    }