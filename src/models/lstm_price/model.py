"""BTC fiyat/getiri tahmini icin LSTM mimarisi."""
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(window: int, num_features: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(window, num_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),  # tek sayisal cikti: bir sonraki log-getiri tahmini
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
