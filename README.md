# Pulsarai-density-mapper

# deploy on Streamlit Community Cloud
- make changes on your local, commit and pust changes to https://github.com/swathi-veytel/Pulsarai-density-mapper
- Go to streamlit app https://pulsarai-density-mapper.streamlit.app/ 
- Click on Manage App on bottom-right corner and select reboot. to deploy changes.

# deploy on docker local
docker build -t app .
docker run -p 8080:8080 app

# Run locally
APP_ENV=LOCAL streamlit run app.py
