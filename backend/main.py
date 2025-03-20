import mlflow
from dotenv import load_dotenv
load_dotenv()

from model_deployment import deploy_model
from inference import get_summary
from evaluation import evaluate_summary
from logging_config import setup_logger, setup_tracing

def run_pipeline(article_text: str, reference_summary: str):
    mlflow.set_experiment("LLMOps_Evaluation")
    with mlflow.start_run():
        logger = setup_logger()
        tracer = setup_tracing()

        try:
            logger.log_text("Starting LLMOps pipeline.")
        except Exception as e:
            print("Warning: Logging failed:", e)
        
        endpoint = deploy_model()
        
        with tracer.start_as_current_span("model_inference", attributes={"model": "bart-large-cnn"}):
            generated_summary = get_summary(endpoint, article_text)
            try:
                logger.log_text(f"Generated summary: {generated_summary}")
            except Exception as e:
                print("Warning: Logging failed:", e)

        with tracer.start_as_current_span("evaluation"):
            scores = evaluate_summary(generated_summary, reference_summary)
            try:
                logger.log_text(f"Evaluation scores: {scores}")
            except Exception as e:
                print("Warning: Logging failed:", e)

        mlflow.log_param("article_length", len(article_text))
        mlflow.log_param("summary_length", len(generated_summary))
        return generated_summary, scores

if __name__ == "__main__":
    sample_article = "Sample news article text covering industry trends and technology updates."
    reference = "Reference summary of the sample article."
    
    summary, metrics = run_pipeline(sample_article, reference)
    print("Generated Summary:", summary)
    print("Metrics:", metrics)
