import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import yfinance as yf
import ta
import sys

def main(ticker, interval, period):
    gspc_data = yf.download(
        ticker,
        interval=interval,
        period=period,
        progress=False
    )

    # DataFrame des features
    feature = pd.DataFrame(index = gspc_data.index)

    close_prices  = gspc_data['Close'].squeeze()
    high_prices   = gspc_data['High'].squeeze()
    low_prices    = gspc_data['Low'].squeeze()
    volume_prices = gspc_data['Volume'].squeeze()

    feature['SMA'] = ta.trend.sma_indicator(close_prices, window=14)
    feature['MACD'] = ta.trend.macd(close_prices)
    feature['RSI'] = ta.momentum.rsi(close_prices)
    feature['Close'] = close_prices
    feature['Bollinger_Upper'] = ta.volatility.bollinger_hband(close_prices)
    feature['Bollinger_Lower'] = ta.volatility.bollinger_lband(close_prices)
    feature['ATR'] = ta.volatility.average_true_range(high_prices, low_prices, close_prices)
    feature['ADX'] = ta.trend.adx(high_prices, low_prices, close_prices)
    feature['OBV'] = ta.volume.on_balance_volume(close_prices, volume_prices)

    gspc_data['SMA'] = feature['SMA']
    gspc_data['MACD'] = feature['MACD']
    gspc_data['RSI'] = feature['RSI']
    gspc_data['Bollinger_Upper'] = feature['Bollinger_Upper']
    gspc_data['Bollinger_Lower'] = feature['Bollinger_Lower']
    gspc_data['ATR'] = feature['ATR']
    gspc_data['ADX'] = feature['ADX']
    gspc_data['OBV'] = feature['OBV']

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(gspc_data['Close'], label='Prix de clôture')
    plt.plot(gspc_data['SMA'], label='SMA (14 périodes)', linestyle='--')
    plt.title(f'Prix de clôture et SMA ({ticker})')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde du graphique en PNG
    plot_path = f"data/{ticker}_{interval}_{period}_plot.png"
    os.makedirs("data", exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    # Nettoyage
    feature.dropna(inplace=True)
    gspc_data.dropna(inplace=True)

    features = feature.drop(columns=['Close'])
    target   = feature['Close']

    # StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA
    pca = PCA(n_components=0.9)
    features_pca = pca.fit_transform(features_scaled)
    
    df_pca = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
    df_final = pd.concat([df_pca, target.reset_index(drop=True)], axis=1)

    output_filename = f"{ticker}_{interval}_{period}.csv"
    output_path = os.path.join("data", output_filename)

    df_final.to_csv(output_path, index=False)
    print(f"Fichier CSV généré : {output_path}")
    print(f"Graphique sauvegardé : {plot_path}")

# interval : [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
if __name__ == "__main__":
    # Récupération des paramètres en CLI (ex: python notebook1.py --ticker ^GSPC --interval 5m --period 1mo)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^GSPC", type=str)
    parser.add_argument("--interval", default="5m", type=str)
    parser.add_argument("--period", default="1mo", type=str)
    args = parser.parse_args()

    main(args.ticker, args.interval, args.period)
