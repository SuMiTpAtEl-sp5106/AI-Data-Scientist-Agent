import streamlit as st
import requests
import os 

st.set_page_config(page_title="Autonomous Data Scientist", page_icon="🤖", layout="wide")

st.title("🤖 Autonomous Agentic Data Scientist Dashboard")
st.markdown("Upload any tabular dataset, specify your target column, and watch your multi-agent LangGraph system clean, engineer, train, and report automatically!")

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    uploaded_file = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx"])
    target_column = st.text_input("Target Column Name", placeholder="e.g., Churn, Survived, median_house_value")
    run_btn = st.button("🚀 Run Autonomous Pipeline", type="primary")

if run_btn:
    if not uploaded_file or not target_column:
        st.error("Please upload a dataset and provide the target column name.")
    else:
        with st.spinner("🤖 Autonomous agents are at work: EDA -> Feature Engineering -> Model Tuning -> Reporting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"target_column": target_column}
            
            try:
                response = requests.post("http://localhost:8000/run-pipeline", files=files, data=data)
                if response.status_code == 200:
                    res_json = response.json()
                    if res_json.get("status") == "success":
                        st.success("🎉 Pipeline Completed Successfully!")
                        
                        champion_model = res_json.get("best_model", res_json.get("best_model_name", "Model"))
                        score = res_json.get("score", res_json.get("best_model_score", 0.0))
                        task_type = res_json.get("task_type", "Classification")
                        
                      
                        metric_label = "R² Score" if task_type == "Regression" else " Accuracy"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Champion Model", champion_model)
                        with col2:
                            st.metric(f"Performance Score ({metric_label})", f"{score:.4f}")
                            
                        st.markdown("---")
                        st.subheader("📁 Executive Summary Report")
                        st.markdown(res_json.get("report", "No report found."))
                    else:
                        st.error(f"Pipeline Error: {res_json.get('message')}")
                else:
                    st.error(f"Server error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend: {e}")


model_path = "storage/best_model.pkl"
if os.path.exists(model_path):
    st.markdown("---")
    st.subheader("📦 Model Export")
    with open(model_path, "rb") as model_file:
        st.download_button(
            label="📥 Download Champion Model (.pkl)",
            data=model_file,
            file_name="champion_model.pkl",
            mime="application/octet-stream",
            help="Download the trained scikit-learn model artifact for production deployment."
        )