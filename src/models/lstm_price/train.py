"""
LSTM fiyat modelini egitir, kaydeder ve test seti uzerinde
degerlendirir.

Kullanim:
    python -m src.models.lstm_price.train
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src import config
from src.models.lstm_price.dataset import prepare_dataset
from src.models.lstm_price.model import build_lstm_model

MODEL_PATH = config.MODEL_DIR / "lstm_price.keras"
SCALER_PATH = config.MODEL_DIR / "lstm_price_scaler.joblib"
HISTORY_PLOT_PATH = config.MODEL_DIR / "lstm_price_training_history.png"
PRED_PLOT_PATH = config.MODEL_DIR / "lstm_price_test_predictions.png"


def return_to_price(prev_close: np.ndarray, predicted_log_return: np.ndarray) -> np.ndarray:
    """Tahmin edilen log-getiriyi, bir onceki kapanis fiyatina uygulayarak
    sayisal bir fiyat tahminine cevirir."""
    return prev_close * np.exp(predicted_log_return)


def evaluate(model, split) -> dict:
    y_pred = model.predict(split.X_test, verbose=0).flatten()

    mae_return = float(np.mean(np.abs(y_pred - split.y_test)))
    rmse_return = float(np.sqrt(np.mean((y_pred - split.y_test) ** 2)))

    # yon dogrulugu: model getirinin isaretini (yukselecek/dusecek) dogru tahmin etti mi
    directional_acc = float(np.mean(np.sign(y_pred) == np.sign(split.y_test)))

    pred_price = return_to_price(split.close_test, y_pred)
    actual_price = return_to_price(split.close_test, split.y_test)
    mae_price = float(np.mean(np.abs(pred_price - actual_price)))

    metrics = {
        "mae_log_return": mae_return,
        "rmse_log_return": rmse_return,
        "directional_accuracy": directional_acc,
        "mae_price_usdt": mae_price,
    }

    plt.figure(figsize=(12, 5))
    plt.plot(actual_price, label="gercek fiyat", linewidth=1)
    plt.plot(pred_price, label="tahmin edilen fiyat", linewidth=1)
    plt.title("Test seti: gercek vs tahmin edilen BTC fiyati")
    plt.xlabel("mum (5dk)")
    plt.ylabel("fiyat (USDT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRED_PLOT_PATH)
    plt.close()

    return metrics


def main():
    print(
        f"[train] veri hazirlaniyor... "
        f"(include_macro={config.INCLUDE_MACRO}, include_xaut={config.INCLUDE_XAUT}, "
        f"include_calendar={config.INCLUDE_CALENDAR})"
    )
    split = prepare_dataset(
        include_macro=config.INCLUDE_MACRO,
        include_xaut=config.INCLUDE_XAUT,
        include_calendar=config.INCLUDE_CALENDAR,
    )
    print(
        f"[train] train={len(split.X_train)}  val={len(split.X_val)}  "
        f"test={len(split.X_test)}  window={split.X_train.shape[1]}  "
        f"features={split.X_train.shape[2]}"
    )

    model = build_lstm_model(window=split.X_train.shape[1], num_features=split.X_train.shape[2])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    joblib.dump(split.scaler, SCALER_PATH)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Egitim gecmisi")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close()

    metrics = evaluate(model, split)
    print("[train] test metrikleri:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")

    print(f"[train] model kaydedildi -> {MODEL_PATH}")
    print(f"[train] scaler kaydedildi -> {SCALER_PATH}")
    print(f"[train] grafikler -> {HISTORY_PLOT_PATH}, {PRED_PLOT_PATH}")


if __name__ == "__main__":
    main()
