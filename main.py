# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

class Params(BaseModel):
    ticker: str
    interval: str
    period: str

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API FastAPI pour la prédiction S&P 500"}

@app.post("/fetch_data")
def fetch_data(params: Params):
    """
    Lance le code du notebook1.py pour générer le CSV et le graphique.
    S'il existe déjà un CSV (et un graphique) pour ce ticker/interval/period,
    on ne refait pas le téléchargement.
    """
    ticker = params.ticker
    interval = params.interval
    period = params.period

    # Vérifier l'existence du CSV
    csv_path = f"data/{ticker}_{interval}_{period}.csv"
    plot_path = f"graphics/{ticker}_{interval}_{period}_plot.png"

    if os.path.exists(csv_path) and os.path.exists(plot_path):
        # Déjà téléchargé
        return {
            "already_fetched": True,
            "plot_path": plot_path,
            "csv_path": csv_path
        }
    else:
        # Lancer le notebook1.py pour télécharger et générer
        cmd = [
            "python", "notebook1.py",
            "--ticker", ticker,
            "--interval", interval,
            "--period", period
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "already_fetched": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "plot_path": plot_path,
            "csv_path": csv_path
        }

@app.post("/predict")
def predict(params: Params):
    """
    Lance le code du notebook2.py pour entraîner le modèle et afficher les prédictions.
    S'il existe déjà un modèle et des graphiques pour ces paramètres, on ne refait pas l'entraînement.
    """
    ticker = params.ticker
    interval = params.interval
    period = params.period

    model_path = f"./models/{ticker}_{interval}_{period}_model.h5"
    train_plot_path = f"graphics/{ticker}_{interval}_{period}_train_plot.png"
    test_plot_path = f"graphics/{ticker}_{interval}_{period}_test_plot.png"

    # Si le modèle existe déjà, on suppose que l'entraînement a déjà été fait
    # et que les deux graphiques existent également.
    if os.path.exists(model_path) and os.path.exists(train_plot_path) and os.path.exists(test_plot_path):
        return {
            "already_trained": True,
            "train_plot_path": train_plot_path,
            "test_plot_path": test_plot_path
        }
    else:
        cmd = [
            "python", "notebook2.py",
            "--ticker", ticker,
            "--interval", interval,
            "--period", period
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "already_trained": False,
            "train_plot_path": train_plot_path,
            "test_plot_path": test_plot_path
        }
