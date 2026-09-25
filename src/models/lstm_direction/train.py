"""
BTC yon (yukari/asagi) siniflandirma modelini egitir, kaydeder ve
degerlendirir.

Kullanim:
    python -m src.models.lstm_direction.train
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src import config
from src.models.lstm_direction.dataset import prepare_direction_dataset
from src.models.lstm_direction.model import build_lstm_classifier

MODEL_PATH = config.MODEL_DIR / "lstm_direction.keras"
SCALER_PATH = config.MODEL_DIR / "lstm_direction_scaler.joblib"
HISTORY_PLOT_PATH = config.MODEL_DIR / "lstm_direction_training_history.png"


def evaluate(model, split) -> dict:
    y_prob = model.predict(split.X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = split.y_test.astype(int)

    accuracy = float(np.mean(y_pred == y_true))

    # SAF COGUNLUK BASARISI: test setinde en cok gorulen sinifi (hep
    # "yukari" ya da hep "asagi") tahmin etseydik ne kadar dogru olurduk?
    # Modelin bunun UZERINE bir sey ogrenip ogrenmedigini gormek icin lazim.
    up_ratio = float(np.mean(y_true))
    majority_baseline = max(up_ratio, 1 - up_ratio)

    # kafa karistirma matrisinin bilesenleri (precision/recall - modelin
    # sadece "hep yukari de" gibi bir kestirmeye kacip kacmadigini gormek icin)
    predicted_up = y_pred == 1
    actual_up = y_true == 1
    true_positive = int(np.sum(predicted_up & actual_up))
    predicted_up_count = int(np.sum(predicted_up))
    actual_up_count = int(np.sum(actual_up))
    precision_up = true_positive / predicted_up_count if predicted_up_count > 0 else float("nan")
    recall_up = true_positive / actual_up_count if actual_up_count > 0 else float("nan")

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # test setinde tek sinif kalmissa (cok kucuk veri) AUC tanimsizdir
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "majority_class_baseline": majority_baseline,
        "up_ratio_in_test": up_ratio,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "auc": auc,
        "predicted_up_ratio": float(np.mean(y_pred)),
    }


def main():
    print(
        f"[train-direction] veri hazirlaniyor... "
        f"(include_macro={config.INCLUDE_MACRO}, include_xaut={config.INCLUDE_XAUT}, "
        f"include_calendar={config.INCLUDE_CALENDAR})"
    )
    split = prepare_direction_dataset(
        include_macro=config.INCLUDE_MACRO,
        include_xaut=config.INCLUDE_XAUT,
        include_calendar=config.INCLUDE_CALENDAR,
    )
    print(
        f"[train-direction] train={len(split.X_train)}  val={len(split.X_val)}  "
        f"test={len(split.X_test)}  window={split.X_train.shape[1]}  "
        f"features={split.X_train.shape[2]}"
    )

    model = build_lstm_classifier(window=split.X_train.shape[1], num_features=split.X_train.shape[2])
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
    plt.title("Egitim gecmisi (yon siniflandirma)")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close()

    metrics = evaluate(model, split)
    print("[train-direction] test metrikleri:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")

    edge = metrics["accuracy"] - metrics["majority_class_baseline"]
    print(f"[train-direction] saf cogunluk tahminine gore fark (edge): {edge:+.6f}")
    if edge <= 0:
        print(
            "[train-direction] UYARI: model, test setinde sadece en sik gorulen "
            "yonu tahmin etmekten daha iyi degil - anlamli bir sinyal yok."
        )

    print(f"[train-direction] model kaydedildi -> {MODEL_PATH}")
    print(f"[train-direction] scaler kaydedildi -> {SCALER_PATH}")
    print(f"[train-direction] grafik -> {HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
