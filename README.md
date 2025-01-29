# Pulsarai-density-mapper

# deploy on cloud
gcloud init
gcloud app deploy

# deploy on docker local
docker build -t app .
docker run -p 8080:8080 app

docker build -t density-mapper .
docker run -d -p 8501:8501 -v /path/to/keys:/app/keys:ro --name density-mapper density-mapper

# Run locally
APP_ENV=LOCAL streamlit run app.py
