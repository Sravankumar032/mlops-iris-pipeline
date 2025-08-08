FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Add working directory to PYTHONPATH
ENV PYTHONPATH=/app

# Run the FastAPI app from api/app/main.py
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
