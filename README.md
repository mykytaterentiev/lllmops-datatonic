# LLMOps Pipeline for Evaluating Generative Text Quality

This project implements a complete LLMOps pipeline that:
- Ingests news articles,
- Deploys a summarization model (using Vertex AI and Hugging Face),
- Generates summaries,
- Evaluates summary quality using ROUGE metrics,
- Logs and traces execution,
- Provides an interactive Streamlit frontend with Plotly visualizations, and
- Implements CI/CD with GitHub Actions.

See [setup_instructions.md](setup_instructions.md) for detailed setup and deployment steps.
