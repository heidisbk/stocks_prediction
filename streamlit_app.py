# streamlit_app.py
import streamlit as st
import requests
import os
from PIL import Image

st.title("Prédiction S&P 500 et autres actifs")

default_ticker = "^GSPC"
default_period = "1mo"

# Liste des intervalles possibles
interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

# Champs de saisie
ticker = st.text_input("TICKER", value=default_ticker)
interval = st.selectbox("INTERVAL", options=interval_options, index=2)  # index=2 => "5m" par défaut
# period = st.text_input("PERIOD", value=default_period)
period = st.selectbox("PERIOD", options=period_options, index=2)

# Pour stocker l'état "data_fetched"
if "data_fetched" not in st.session_state:
    st.session_state["data_fetched"] = False

fastapi_url = "http://localhost:8000"  # adapter si besoin

# --------------------------------------------------------------------------------------
# 1) Bouton pour récupérer les données
# --------------------------------------------------------------------------------------
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

        # Si c'est déjà fetché, on ne ré-affiche pas stdout/stderr
        if resp_data.get("already_fetched"):
            st.write("Les données étaient déjà présentes en local, pas de nouveau téléchargement.")
        else:
            st.write("Téléchargement et génération du CSV terminés.")
            # st.write("Sortie standard :", resp_data.get("stdout", ""))
            # st.write("Erreurs :", resp_data.get("stderr", ""))

        plot_path = resp_data["plot_path"]

        # Affichage du graphique Notebook 1
        if os.path.exists(plot_path):
            image = Image.open(plot_path)
            st.image(image, caption=f"Graphique - {ticker} ({interval}, {period})")
            # On marque que les données sont disponibles
            st.session_state["data_fetched"] = True
        else:
            st.warning(f"Fichier introuvable : {plot_path}")
            st.session_state["data_fetched"] = False
    else:
        st.error(f"Erreur : {response.status_code}")
        st.session_state["data_fetched"] = False

# --------------------------------------------------------------------------------------
# 2) Bouton pour lancer la prédiction (uniquement si data_fetched est True)
# --------------------------------------------------------------------------------------
if st.session_state["data_fetched"]:
    # On affiche le bouton sous le premier graphique
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

            if resp_data.get("already_trained"):
                st.write("Le modèle était déjà entraîné pour ces paramètres. Pas de nouvel entraînement.")
            else:
                st.write("Entraînement terminé (ou ré-entraînement).")

            train_plot_path = resp_data["train_plot_path"]
            test_plot_path  = resp_data["test_plot_path"]

            # Affichage des deux graphiques
            if os.path.exists(train_plot_path):
                image_train = Image.open(train_plot_path)
                st.image(image_train, caption="Graphique d'entraînement")
            else:
                st.warning(f"Fichier introuvable : {train_plot_path}")

            if os.path.exists(test_plot_path):
                image_test = Image.open(test_plot_path)
                st.image(image_test, caption="Graphique de test")
            else:
                st.warning(f"Fichier introuvable : {test_plot_path}")

        else:
            st.error(f"Erreur : {response.status_code}")
