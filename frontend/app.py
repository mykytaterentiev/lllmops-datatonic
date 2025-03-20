import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.graph_objects as go
from google.cloud import aiplatform
from backend.inference import get_summary
from backend.evaluation import evaluate_summary

PROJECT_ID = os.getenv("PROJECT_ID", "default-project-id")
REGION = os.getenv("REGION", "europe-west8")
aiplatform.init(project=PROJECT_ID, location=REGION)

endpoint_bart = aiplatform.Endpoint(endpoint_name="projects/276261586056/locations/europe-west8/endpoints/4248046736796286976")
endpoint_t5 = aiplatform.Endpoint(endpoint_name="projects/276261586056/locations/europe-west8/endpoints/4248046736796286976")

st.title("LLMOps Pipeline for Evaluating Generative Text Quality")
st.write("Provide a news article and choose the summarization models for evaluation.")

article_text = st.text_area("News Article Input:", height=200)

models_to_evaluate = st.multiselect("Select Models:", ["BART", "T5"])

if st.button("Run Evaluation") and article_text and models_to_evaluate:
    results = {}
    reference_summary = "This is a reference summary for quality evaluation."

    if "BART" in models_to_evaluate:
        bart_summary = get_summary(endpoint_bart, article_text)
        bart_scores = evaluate_summary(bart_summary, reference_summary)
        results["BART"] = {"summary": bart_summary, "scores": bart_scores}

    if "T5" in models_to_evaluate:
        t5_summary = get_summary(endpoint_t5, article_text)
        t5_scores = evaluate_summary(t5_summary, reference_summary)
        results["T5"] = {"summary": t5_summary, "scores": t5_scores}

    st.subheader("Evaluation Results")
    for model, data in results.items():
        st.markdown(f"### Model: {model}")
        st.write("**Generated Summary:**")
        st.info(data['summary'])
        st.write("**ROUGE Evaluation Scores:**")
        st.json(data['scores'])
        
        rouge_scores = data['scores'][0]
        labels = list(rouge_scores.keys())
        f1_values = [rouge_scores[label]['f'] for label in labels]
        
        fig = go.Figure(data=[go.Bar(x=labels, y=f1_values)])
        fig.update_layout(title=f"{model} ROUGE F1 Scores", xaxis_title="Metric", yaxis_title="F1 Score")
        st.plotly_chart(fig)