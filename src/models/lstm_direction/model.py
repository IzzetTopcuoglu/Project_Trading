"""BTC yon (yukari/asagi) siniflandirmasi icin LSTM mimarisi.

lstm_price/model.py'deki ile ayni gizli katman yapisi (64+32 LSTM unit)
kullanilir, tek fark cikis katmani: sigmoid aktivasyonlu tek noron
(P(yukari) olasiligini 0-1 arasinda verir) ve binary cross-entropy loss
(dogrudan siniflandirma hatasini optimize eder, MSE degil).
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_classifier(window: int, num_features: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(window, num_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # P(bir sonraki mum yukari kapanir)
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
