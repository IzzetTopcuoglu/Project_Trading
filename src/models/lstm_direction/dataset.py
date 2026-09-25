"""
BTC yon tahmini (siniflandirma) icin veri hazirlama.

lstm_price/dataset.py'deki ile AYNI ozellik muhendisligini (FEATURE_COLUMNS,
MACRO_FEATURE_COLUMNS, add_features, add_macro_features) tekrar kullanir -
tek fark HEDEF DEGISKEN: burada bir sonraki mumun log-getirisinin SAYISAL
degeri degil, sadece YONU (yukari=1 / asagi=0) tahmin edilir.

Neden ayri bir model/paket (lstm_price'i degistirmek yerine)?
  - lstm_price'daki regresyon deneyleri (v1-v3) ve README'deki sonuclar
    halen gecerli/karsilastirilabilir kalsin diye dokunulmadi.
  - Bu, "kervanı yolda düzmek" yaklasimiyla uyumlu: yeni bir deney,
    eskisini bozma riski olmadan yan yana eklendi.

Neden siniflandirma (regresyon + isaretine bakmak yerine)?
  - Regresyon modeli MSE'yi (sayisal hata) optimize ediyor, ama bizim asil
    onemsedigimiz sey YÖN. Model direkt "yukari/asagi" olasiligini tahmin
    edecek sekilde (sigmoid + binary cross-entropy) egitilirse, tam olcmek
    istedigimiz metrigi (directional_accuracy) optimize etmis oluruz.

ONEMLI degerlendirme notu: test setinde "yukari" mumlarin orani %50'den
farkliysa (ornegin %53 yukari, %47 asagi), model hicbir sey ogrenmeden
surekli "yukari" tahmin ederek bile o orana yakin bir "dogruluk"
gosterebilir. Bu yuzden train.py, sonucu her zaman bu SAF COGUNLUK
BASARISI ile karsilastirir - modelin gercekten bir sey ogrenip
ogrenmedigini anlamak icin.
"""
import numpy as np
import pandas as pd

from src.models.lstm_price.dataset import (  # noqa: F401 (re-exported)
    FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    XAUT_FEATURE_COLUMNS,
    CALENDAR_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_features,
    add_macro_features,
    add_xaut_feature,
    add_calendar_features,
    load_ohlcv,
    time_based_split,
)
from src import config
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

DIRECTION_COLUMN = "target_direction"


@dataclass
class DirectionDatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def build_sequences(features: np.ndarray, targets: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(features) - window + 1):
        X.append(features[i : i + window])
        y.append(targets[i + window - 1])
    return np.array(X), np.array(y)


def prepare_direction_dataset(
    window: int = None,
    include_macro: bool = False,
    include_xaut: bool = False,
    include_calendar: bool = False,
) -> DirectionDatasetSplit:
    window = window or config.WINDOW

    df = load_ohlcv()
    df = add_features(df)

    # yon etiketi: bir sonraki mumun getirisi pozitif mi (1) negatif/sifir mi (0)
    df[DIRECTION_COLUMN] = (df[TARGET_COLUMN] > 0).astype(np.float32)

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

    train_df, val_df, test_df = time_based_split(df)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].values)

    def _scaled(d):
        return scaler.transform(d[feature_columns].values)

    X_train, y_train = build_sequences(_scaled(train_df), train_df[DIRECTION_COLUMN].values, window)
    X_val, y_val = build_sequences(_scaled(val_df), val_df[DIRECTION_COLUMN].values, window)
    X_test, y_test = build_sequences(_scaled(test_df), test_df[DIRECTION_COLUMN].values, window)

    return DirectionDatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )
