import streamlit as st
import requests
import os
from PIL import Image

st.title("Interface Web - Prédiction S&P 500 (Streamlit + FastAPI)")

default_ticker = "^GSPC"
default_interval = "5m"
default_period = "1mo"

ticker = st.text_input("TICKER", value=default_ticker)
interval = st.text_input("INTERVAL", value=default_interval)
period = st.text_input("PERIOD", value=default_period)

fastapi_url = "http://localhost:8000"  # si vous êtes en local

# Bouton pour récupérer les données
if st.button("Récupérer les données"):
    st.write("En cours de récupération des données...")
    payload = {
        "ticker": ticker,
        "interval": interval,
        "period": period
    }
    response = requests.post(f"{fastapi_url}/fetch_data", json=payload)
    if response.status_code == 200:
        resp_data = response.json()
        st.write("Sortie standard :", resp_data["stdout"])
        st.write("Erreurs :", resp_data["stderr"])

        # Affichage du graphique
        plot_path = resp_data["plot_path"]
        if os.path.exists(plot_path):
            image = Image.open(plot_path)
            st.image(image, caption="Graphique - Notebook 1")
        else:
            st.warning(f"Fichier introuvable : {plot_path}")
    else:
        st.error(f"Erreur : {response.status_code}")

# Bouton pour lancer la prédiction
if st.button("Faire une prédiction"):
    st.write("En cours de prédiction...")
    payload = {
        "ticker": ticker,
        "interval": interval,
        "period": period
    }
    response = requests.post(f"{fastapi_url}/predict", json=payload)
    if response.status_code == 200:
        resp_data = response.json()
        st.write("Sortie standard :", resp_data["stdout"])
        st.write("Erreurs :", resp_data["stderr"])

        # Affichage des deux graphiques
        train_plot_path = resp_data["train_plot_path"]
        test_plot_path  = resp_data["test_plot_path"]

        if os.path.exists(train_plot_path):
            image_train = Image.open(train_plot_path)
            st.image(image_train, caption="Graphique d'entraînement - Notebook 2")
        else:
            st.warning(f"Fichier introuvable : {train_plot_path}")

        if os.path.exists(test_plot_path):
            image_test = Image.open(test_plot_path)
            st.image(image_test, caption="Graphique de test - Notebook 2")
        else:
            st.warning(f"Fichier introuvable : {test_plot_path}")

    else:
        st.error(f"Erreur : {response.status_code}")
