import mlflow
from rouge import Rouge

def evaluate_summary(generated_summary, reference_summary):
    """
    Compute ROUGE metrics for the generated summary against a reference summary.
    Ensures both inputs are strings.
    """
    generated_summary = str(generated_summary)
    reference_summary = str(reference_summary)
    
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    
    mlflow.log_metric("rouge_1_f", scores[0]['rouge-1']['f'])
    mlflow.log_metric("rouge_2_f", scores[0]['rouge-2']['f'])
    mlflow.log_metric("rouge_l_f", scores[0]['rouge-l']['f'])
    return scores
