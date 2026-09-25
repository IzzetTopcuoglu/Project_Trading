"""
Egitilmis LSTM modeliyle, elindeki en guncel mum verisinden bir
sonraki mumun fiyatini tahmin eder.

Kullanim:
    python -m src.models.lstm_price.predict
"""
import joblib
import numpy as np
from tensorflow import keras

from src import config
from src.models.lstm_price.dataset import (
    FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    XAUT_FEATURE_COLUMNS,
    CALENDAR_FEATURE_COLUMNS,
    add_features,
    add_macro_features,
    add_xaut_feature,
    add_calendar_features,
    load_ohlcv,
)
from src.models.lstm_price.train import MODEL_PATH, SCALER_PATH, return_to_price


def predict_next_price(
    window: int = None,
    include_macro: bool = None,
    include_xaut: bool = None,
    include_calendar: bool = None,
) -> dict:
    window = window or config.WINDOW
    include_macro = config.INCLUDE_MACRO if include_macro is None else include_macro
    include_xaut = config.INCLUDE_XAUT if include_xaut is None else include_xaut
    include_calendar = config.INCLUDE_CALENDAR if include_calendar is None else include_calendar

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = add_features(load_ohlcv())
    feature_columns = list(FEATURE_COLUMNS)
    if include_macro:
        df = add_macro_features(df)
        df = df.dropna(subset=MACRO_FEATURE_COLUMNS).reset_index(drop=True)
        feature_columns = feature_columns + MACRO_FEATURE_COLUMNS
    if include_xaut:
        df = add_xaut_feature(df)
        df = df.dropna(subset=XAUT_FEATURE_COLUMNS).reset_index(drop=True)
        feature_columns = feature_columns + XAUT_FEATURE_COLUMNS
    if include_calendar:
        df = add_calendar_features(df)
        feature_columns = feature_columns + CALENDAR_FEATURE_COLUMNS

    last_window = df.tail(window)
    if len(last_window) < window:
        raise ValueError(f"Yeterli veri yok: {len(last_window)} < {window}. Once fetch_data.py calistir.")

    X = scaler.transform(last_window[feature_columns].values)
    X = X.reshape(1, window, len(feature_columns))

    predicted_log_return = float(model.predict(X, verbose=0).flatten()[0])
    last_close = float(last_window["close"].iloc[-1])
    predicted_price = float(return_to_price(last_close, predicted_log_return))

    return {
        "last_close": last_close,
        "predicted_log_return": predicted_log_return,
        "predicted_price": predicted_price,
        "direction": "yukselis" if predicted_log_return > 0 else "dusus",
    }


if __name__ == "__main__":
    result = predict_next_price()
    print("[predict]", result)
