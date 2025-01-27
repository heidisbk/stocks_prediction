import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Add
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

def transformer_block(inputs, model_dim, num_heads, ff_dim, dropout=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(model_dim)(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff_output)
    return out2

def positional_encoding(max_position, model_dim):
    angle_rads = np.arange(max_position)[:, np.newaxis] / np.power(
        10000,
        (2 * (np.arange(model_dim)[np.newaxis, :] // 2)) / np.float32(model_dim)
    )
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim, dropout=0.1):
    inputs = Input(input_shape)
    x = Dense(model_dim)(inputs)  # Projection initiale
    max_position = input_shape[0]
    pe = positional_encoding(max_position, model_dim)
    x = x + pe
    for _ in range(num_layers):
        x = transformer_block(x, model_dim, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def direction_accuracy(y_true, y_pred):
    y_true_shift = y_true[1:] - y_true[:-1]
    y_pred_shift = y_pred[1:] - y_pred[:-1]
    length = tf.minimum(tf.shape(y_true_shift)[0], tf.shape(y_pred_shift)[0])
    y_true_shift = y_true_shift[:length]
    y_pred_shift = y_pred_shift[:length]
    correct_dir = tf.equal(tf.sign(y_true_shift), tf.sign(y_pred_shift))
    return tf.reduce_mean(tf.cast(correct_dir, tf.float32))

def main(ticker, interval, period):
    filename = f"./data/{ticker}_{interval}_{period}.csv"
    if not os.path.exists(filename):
        print(f"Le fichier {filename} n'existe pas. Veuillez d'abord exécuter notebook1.py.")
        return

    df = pd.read_csv(filename)
    print("Aperçu du DataFrame chargé :")
    print(df.head())

    mean_dict = {}
    std_dict = {}
    for col in df.columns:
        mean_dict[col] = df[col].mean()
        std_dict[col] = df[col].std()

    X = df.drop(columns=['Close'])
    y = df['Close']

    TRAIN_DATA_RATIO = 0.8
    train_data_size = int(len(X) * TRAIN_DATA_RATIO)

    train = X[:train_data_size]
    test = X[train_data_size:]

    NUMBER_OF_SERIES_FOR_PREDICTION = 24

    def create_dataset(dataset, number_of_series_for_prediction=24):
        X_data, y_data = [], []
        data_np = dataset.to_numpy()
        
        for i in range(len(data_np) - number_of_series_for_prediction):
            X_data.append(data_np[i : i + number_of_series_for_prediction])
            y_data.append(data_np[i + number_of_series_for_prediction, -1])  # On vise la dernière colonne
        return np.array(X_data), np.array(y_data)

    X_train, y_train = create_dataset(train, NUMBER_OF_SERIES_FOR_PREDICTION)
    X_test, y_test = create_dataset(test, NUMBER_OF_SERIES_FOR_PREDICTION)

    print("Forme des données :")
    print(f"X_train : {X_train.shape}, y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}, y_test  : {y_test.shape}")

    model_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 256
    output_dim = 1

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim, dropout=0.1)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=MeanAbsoluteError(),
        metrics=[direction_accuracy]
    )

    model.summary()

    def custom_lr_schedule(epoch, lr):
        warmup_epochs = 10
        warmup_lr = 1e-4
        initial_lr = 1e-3
        decay_rate = 0.4
        decay_step = 10

        if epoch < warmup_epochs:
            lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / warmup_epochs)
        else:
            lr = initial_lr * (decay_rate ** ((epoch - warmup_epochs) / decay_step))
        return lr

    lr_scheduler = LearningRateScheduler(custom_lr_schedule)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        min_delta=1e-4,
        mode='min',
        restore_best_weights=True
    )

    os.makedirs("./models", exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath='./models/model_checkpoint.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [lr_scheduler, early_stopping, model_checkpoint, reduce_lr]

    num_epochs = 10
    batch_size = 64

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    eval_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nRésultats du test : {model.metrics_names}")
    print(eval_results)

    # Prédictions sur le train
    train_preds = model.predict(X_train)
    train_preds_rescaled = train_preds * std_dict['Close'] + mean_dict['Close']
    y_train_rescaled = y_train * std_dict['Close'] + mean_dict['Close']

    plt.figure(figsize=(12, 6))
    plt.plot(y_train_rescaled, label='Vraies valeurs (Train)', color='blue')
    plt.plot(train_preds_rescaled, label='Prédictions (Train)', color='red', linestyle='--')
    plt.title('Train : Comparaison Vrai/Predict')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)

    train_plot_path = f"data/{ticker}_{interval}_{period}_train_plot.png"
    plt.savefig(train_plot_path)
    plt.close()

    # Prédictions sur le test
    test_preds = model.predict(X_test)
    test_preds_rescaled = test_preds * std_dict['Close'] + mean_dict['Close']
    y_test_rescaled = y_test * std_dict['Close'] + mean_dict['Close']

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Vraies valeurs (Test)', color='blue')
    plt.plot(test_preds_rescaled, label='Prédictions (Test)', color='red', linestyle='--')
    plt.title('Test : Comparaison Vrai/Predict')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)

    test_plot_path = f"data/{ticker}_{interval}_{period}_test_plot.png"
    plt.savefig(test_plot_path)
    plt.close()

    model.save('./models/transformer_final.h5')
    print("Modèle sauvegardé : ./models/transformer_final.h5")
    print(f"Graphique d'entraînement : {train_plot_path}")
    print(f"Graphique de test : {test_plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^GSPC", type=str)
    parser.add_argument("--interval", default="5m", type=str)
    parser.add_argument("--period", default="1mo", type=str)
    args = parser.parse_args()

    main(args.ticker, args.interval, args.period)
