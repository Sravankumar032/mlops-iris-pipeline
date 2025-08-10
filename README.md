# mlops-iris-pipeline
Assignment1: Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices

# Execution Steps and Important Commands
# GIT Commands
git add . -f
git commit -m "Commit After enabling Monitor & logging part"
git push origin main

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Train model
python -m src.models.train_model

# mlflow
mlflow ui --backend-store-uri artifacts/mlruns
Local Host: http://127.0.0.1:5000

# Build Docker Image
docker build -t iris-api .
docker rm -f mlops-api
docker build -t mlops-api:latest .

# Run Docker Container (Expose API)
docker run -p 8000:8000 iris-api
docker run -d -p 8000:8000 --name mlops-api mlops-api:latest

# Example Test API
https://verbose-capybara-6954pprqggjxc57r9-8000.app.github.dev/docs
https://verbose-capybara-6954pprqggjxc57r9-8000.app.github.dev/metrics

# Input Json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Resources
# DockerHub Link
https://hub.docker.com/repositories/sravanmlops01