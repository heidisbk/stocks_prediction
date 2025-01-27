from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import base64

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
    """Lance le code du notebook1.py pour générer le CSV et le graphique."""
    ticker = params.ticker
    interval = params.interval
    period = params.period

    cmd = [
        "python", "notebook1.py",
        "--ticker", ticker,
        "--interval", interval,
        "--period", period
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Path du graph
    plot_path = f"data/{ticker}_{interval}_{period}_plot.png"
    resp = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "plot_path": plot_path,
        "csv_path": f"data/{ticker}_{interval}_{period}.csv"
    }
    return resp

@app.post("/predict")
def predict(params: Params):
    """Lance le code du notebook2.py pour entraîner le modèle et afficher les prédictions."""
    ticker = params.ticker
    interval = params.interval
    period = params.period

    cmd = [
        "python", "notebook2.py",
        "--ticker", ticker,
        "--interval", interval,
        "--period", period
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    train_plot_path = f"data/{ticker}_{interval}_{period}_train_plot.png"
    test_plot_path = f"data/{ticker}_{interval}_{period}_test_plot.png"

    resp = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "train_plot_path": train_plot_path,
        "test_plot_path": test_plot_path
    }
    return resp
