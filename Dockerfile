FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r backend/requirements.txt && \
    pip install -r frontend/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py"]