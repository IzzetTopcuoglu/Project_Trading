"""
Veriyi/modeli adim adim incelemek icin kesif dosyasi.

Bu dosyayi VS Code'da ac. Her "# %%" ile baslayan blok ayri bir hucre -
icine tikla, sonra Ctrl+Enter (ya da hucrenin ustunde cikan "Run Cell"
yazisina tikla). Ilk calistirdiginda VS Code senden "Jupyter kernel"
secmeni isteyebilir - sanal ortamini (.venv) sec.

Hucreleri SIRAYLA calistir (yukaridan asagiya) - alttaki hucreler
ustteki hucrelerde tanimlanan degiskenleri kullaniyor.

Calistirdiktan sonra ekranin sagindaki "Interactive Window"da hem
ciktiyi gorursun, hem de ustteki arac cubugunda bir "Variables"
("Degiskenler") butonu cikar - ona tiklayip o an tanimli olan TUM
degiskenleri (df, X_train, vb.) bir tablo gibi inceleyebilirsin,
DataFrame'lere cift tiklayinca Excel benzeri bir gorunumde acilirlar.
"""

# %% Kurulum - once bunu calistir
import pandas as pd
import duckdb

from src import config
from src.models.lstm_price.dataset import (
    load_ohlcv,
    add_features,
    load_macro_daily,
    add_macro_features,
    load_xaut_ohlcv,
    add_xaut_feature,
    prepare_dataset,
    FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    XAUT_FEATURE_COLUMNS,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% 1) DuckDB'de hangi tablolar var, kac satir?
con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
print(con.execute("SHOW TABLES").fetchdf())
for t in ["ohlcv_5m", "macro_daily", "xaut_ohlcv"]:
    try:
        n = con.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchdf()["n"][0]
        print(f"{t}: {n} satir")
    except Exception as e:
        print(f"{t}: yok ({e})")
con.close()

# %% 2) Ham BTC verisi neye benziyor? (fetch_data.py'nin ciktisi, hic islenmemis)
raw_btc = load_ohlcv()
print(raw_btc.shape)
raw_btc.head(10)

# %% 3) Ham makro veri (fetch_macro_data.py'nin ciktisi - gunluk)
raw_macro = load_macro_daily()
print(raw_macro.shape)
raw_macro.tail(10)

# %% 4) Ham XAUT verisi (fetch_xaut_data.py'nin ciktisi - BTC ile ayni mum araliginda)
raw_xaut = load_xaut_ohlcv()
print(raw_xaut.shape)
raw_xaut.head(10)

# %% 5) Ozellik muhendisligi SONRASI BTC (log_return, rsi_14, vb. eklenmis hali)
btc_features = add_features(raw_btc)
print(btc_features.shape)
print("feature sutunlari:", FEATURE_COLUMNS)
btc_features[["datetime", "close"] + FEATURE_COLUMNS].head(10)

# %% 6) Makro ozellikleri eklenince ne oluyor (1 gun kaydirilmis log-getiriler)
with_macro = add_macro_features(btc_features)
with_macro[["datetime", "close"] + MACRO_FEATURE_COLUMNS].head(10)

# %% 7) XAUT ozelligi eklenince ne oluyor (ayni anki log-getiri, leakage yok)
with_xaut = add_xaut_feature(btc_features)
with_xaut[["datetime", "close"] + XAUT_FEATURE_COLUMNS].head(10)

# %% 8) Egitime giden NIHAI hal: prepare_dataset() - scaler'a girmeden hemen once
#    (asagida hem include_macro hem include_xaut True - istedigini degistir)
split = prepare_dataset(include_macro=True, include_xaut=True)
print("X_train:", split.X_train.shape, "  (ornek_sayisi, pencere_uzunlugu, ozellik_sayisi)")
print("y_train:", split.y_train.shape)
print("X_val:", split.X_val.shape)
print("X_test:", split.X_test.shape)

# %% 9) Tek bir egitim ornegine (pencereye) yakindan bak
#    X_train[0], ilk 60 mumluk pencere - HER SATIR bir mum, HER SUTUN bir
#    ozellik (scaler'dan gectigi icin degerler standardize edilmis -
#    ortalama ~0, std ~1 civarinda olmali)
ornek_pencere = split.X_train[0]
print(ornek_pencere.shape)  # (window, num_features)
pd.DataFrame(ornek_pencere)

# %% 10) O pencerenin hedefi (bir sonraki mumun log-getirisi)
print("bu pencerenin hedefi (y_train[0]):", split.y_train[0])

# %% 11) Scaler'in ogrendigi ortalama/std degerleri (hangi ozellik ne olcekte)
import numpy as np

# yukaridaki 8. hucrede prepare_dataset'i hangi include_macro/include_xaut
# ile cagirdiysan, feature_columns_used listesini de ona gore ayarla
feature_columns_used = list(FEATURE_COLUMNS) + MACRO_FEATURE_COLUMNS + XAUT_FEATURE_COLUMNS
scaler_info = pd.DataFrame(
    {
        "feature": feature_columns_used,
        "mean": split.scaler.mean_,
        "std": np.sqrt(split.scaler.var_),
    }
)
scaler_info
