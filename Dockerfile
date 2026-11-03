FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraudshield ./fraudshield
COPY streamlit_app.py .
COPY models ./models
COPY data/echantillons.csv ./data/echantillons.csv
COPY data/eda_summary.json ./data/eda_summary.json

EXPOSE 8001 8501

CMD ["python", "-m", "uvicorn", "fraudshield.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
