import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from huggingface_hub import get_token

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "default-project-id")
REGION = os.getenv("REGION", "europe-west8")
HF_MODEL_ID = "facebook/bart-large-cnn"  
HF_TASK = "summarization"
HF_PYTORCH_URI = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-2.transformers.4-44.ubuntu2204.py311"

hf_token = os.getenv("HF_TOKEN") or get_token()

def deploy_model():
    """
    Deploy the Hugging Face model to Vertex AI and return the endpoint.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    model = aiplatform.Model.upload(
        display_name=f"{HF_MODEL_ID.split('/')[-1]}-summarization",
        serving_container_image_uri=HF_PYTORCH_URI,
        serving_container_environment_variables={
            "HF_MODEL_ID": HF_MODEL_ID,
            "HF_TASK": HF_TASK,
            "HF_TOKEN": hf_token,
        },
    )
    print("Creating Endpoint")
    endpoint = aiplatform.Endpoint.create(display_name=f"{model.display_name}-endpoint")
    print("Endpoint created. Resource name:", endpoint.resource_name)
    
    deployed_model = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model.display_name,
        machine_type="n2-standard-4",  
        accelerator_type=None,
        accelerator_count=0,
    )
    print(f"Deploying model to Endpoint: {endpoint.resource_name}")
    print(f"Model deployed to: {deployed_model.resource_name}")
    return endpoint

if __name__ == "__main__":
    deploy_model()