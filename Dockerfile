FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

# Add working directory to PYTHONPATH
ENV PYTHONPATH=/app

# Run the FastAPI app from api/app/main.py
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

COPY artifacts /app/artifacts
