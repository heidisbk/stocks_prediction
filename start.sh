#!/bin/bash

# Lancement du serveur FastAPI (main.py) en arrière-plan
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Lancement de l'app Streamlit (streamlit_app.py) au premier plan
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
