def get_summary(endpoint, article_text: str):
    """
    Invoke the Vertex AI endpoint to generate a summary for the given article_text.
    The model may return the prediction as a dict. We extract the summary string accordingly.
    """
    prediction = endpoint.predict(instances=[article_text])
    summary = prediction.predictions[0]
    
    if isinstance(summary, dict):
        summary = summary.get("generated_text", str(summary))
    
    if isinstance(summary, list):
        summary = " ".join(str(s) for s in summary)
    else:
        summary = str(summary)
    
    return summary
