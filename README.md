# mlops-iris-pipeline
Assignment1: Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices

# Execution Steps and Important Commands
# GIT Commands
git add . -f
git commit -m "Commit After enabling Monitor & logging part"
git push origin main

# Read & Process iris Data
python -m src.data.load_data
python -m src.data.preprocess

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
docker build -t iris-api . (Test)
docker rm -f mlops-api (Test)
docker build -t mlops-api:latest . (main)

# Run Docker Container (Expose API)
docker compose up --build
docker run -p 8000:8000 iris-api (Optional)
docker run -d -p 8000:8000 --name mlops-api mlops-api:latest (Optional)

# Input Json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Other User Full Commands
docker rm -f prometheus
docker rm -f grafana
docker rm -f mlops-api
docker compose down
docker compose up --build

# Resources
GitHub Link: https://hub.docker.com/repositories/sravanmlops01
IRIS Prediction API: https://zany-trout-975jxxqr979739p6v-8000.app.github.dev/docs
API Metrics: https://zany-trout-975jxxqr979739p6v-8000.app.github.dev/metrics
prometheus: https://zany-trout-975jxxqr979739p6v-9090.app.github.dev/
Grafana Dashboard: https://zany-trout-975jxxqr979739p6v-3000.app.github.dev/
Data Source COnnection in Grafana: http://prometheus:9090