"""
BTC fiyat tahmini LSTM'i icin veri hazirlama.

Onemli tasarim kararlari (data leakage'i onlemek icin):
  1. Train/val/test bolme ZAMAN SIRASINA gore yapilir (rastgele shuffle YOK).
     Suffle edilirse model gelecekteki veriyi "gorup" gecmisi tahmin
     etmis gibi yaniltici basari gosterir.
  2. Scaler (StandardScaler) SADECE train bolumune fit edilir, sonra
     val/test'e sadece transform uygulanir. Val/test istatistiklerinin
     scaler'a sizmasi da bir tur leakage'dir.
  3. Model dogrudan fiyati degil, bir sonraki mumun log-getirisini
     (log(close_t+1 / close_t)) tahmin eder. Fiyat serisi durağan
     (stationary) olmadigi icin ham fiyati tahmin etmeye calismak
     modelin sadece "bir onceki fiyati tekrar et" seklinde bir kestirim
     yapmasina yol acar. Egitimden sonra tahmin edilen getiri, guncel
     fiyata uygulanarak sayisal bir fiyat tahminine donusturulur
     (bkz. predict.py:return_to_price).
  4. FEATURE_COLUMNS'daki hicbir ozellik ham fiyat/hacim SEVIYESI
     degildir (v1'de open/high/low/close/volume dogrudan kullaniliyordu,
     bu bir hataydi). BTC fiyati aylar icinde onemli olcude trend
     yapabiliyor (ornegin test setinde 62k -> 82k), bu yuzden train
     doneminde fit edilen bir scaler'a test donemindeki ham fiyat
     degerlerini vermek, modelin egitimde hic gormedigi bir olcekte
     sayilarla karsi karsiya kalmasina yol acar (covariate shift).
     Bunun yerine hepsi oranlar/rolling istatistikler seklinde,
     zamandan bagimsiz (stationary) ozelliklere cevrildi.
  5. Makro veri (fetch_macro_data.py'nin urettigi GUNLUK veri) eklenirken
     iki ayri leakage riskine dikkat edildi:
       a) Makro degerler de (S&P500, altin, vs.) ham seviye olarak
          verilmez - BTC'de yaptigimiz gibi gunluk log-getiriye
          cevrilir, yoksa yine covariate shift sorunu yasariz.
       b) Bir gunun BTC mumlari, O GUNUN DEGIL, BIR ONCEKI GUNUN makro
          kapanis degerlerini gorur (tarihi 1 gun ileri kaydirarak).
          Cunku ornegin S&P500'un "bugunku" kapanisi, ABD borsasi
          kapanana kadar gercekte bilinmiyor - bugunun sabah mumuna
          bugunun kapanisini vermek gelecegi gormek olurdu.
  6. XAUT (Tether Gold - Binance'de 7/24 islem goren altina bagli
     token) icin bu 1-gun-kaydirma GEREKMEZ: XAUT, BTC ile TAM AYNI
     mum araliginda ve ayni anda kapanan bir kripto varlik, o yuzden
     bir mumun xaut_log_return'u o mumun kendi log_return'u ile ayni
     zamanda biliniyor (bkz. add_xaut_feature). Makro veri (gunluk,
     geleneksel piyasa) ile XAUT (surekli, kripto piyasasi) arasindaki
     bu fark, iki farkli birlestirme stratejisi gerektirdi.
"""
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.economic_calendar import add_calendar_features

FEATURE_COLUMNS = [
    "log_return",       # bu mumun getirisi
    "hl_range",         # mum ici oynaklik: (high-low)/close
    "oc_range",         # mum govdesi: (close-open)/open
    "price_to_sma20",   # fiyatin 20 mumluk ortalamaya gore konumu
    "volatility_20",    # son 20 mumun getiri oynakligi (rolling std)
    "rsi_14",           # 14 periyotluk RSI (0-100 arasi, dogal olarak stationary)
    "volume_z",         # hacmin son 20 muma gore z-skoru
]

# fetch_macro_data.py'nin ciktigi HAM sutun adlari -> bunlardan turetilen
# stationary (log-getiri / fark) feature adlari
MACRO_TABLE = "macro_daily"
_MACRO_RETURN_SOURCE_COLUMNS = [
    "approx_total_market_cap",
    "usdt_total_volume",
    "sp500",
    "gold",
    "brent_oil",
    "silver",
]
_MACRO_DIFF_SOURCE_COLUMNS = ["us10y_yield"]  # zaten bir oran, log yerine fark

MACRO_FEATURE_COLUMNS = (
    [f"macro_{c}_ret" for c in _MACRO_RETURN_SOURCE_COLUMNS]
    + [f"macro_{c}_chg" for c in _MACRO_DIFF_SOURCE_COLUMNS]
)

# XAUT (Tether Gold): BTC ile AYNI mum araliginda islem goren, 7/24
# guncellenen altina bagli kripto token - fetch_xaut_data.py tarafindan
# xaut_ohlcv tablosuna yazilir.
XAUT_FEATURE_COLUMNS = ["xaut_log_return"]

# Ekonomik takvim (FOMC/CPI/NFP) - bkz. src/data/economic_calendar.py.
# Orderbook'un aksine bu GECMISE de uygulanabilir (leakage riski yok,
# cunku bu tarihler aylar oncesinden kamuya acik olarak biliniyordu).
CALENDAR_FEATURE_COLUMNS = ["calendar_days_to_next_event", "calendar_high_impact_soon"]

TARGET_COLUMN = "target_log_return"


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # test setinin fiyatlari: tahminleri gercek fiyata cevirip
    # degerlendirmek icin lazim
    close_test: np.ndarray
    scaler: StandardScaler


def load_ohlcv(table_name: str = None) -> pd.DataFrame:
    table_name = table_name or config.OHLCV_TABLE
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    df = con.execute(f"SELECT * FROM {table_name} ORDER BY timestamp").fetchdf()
    con.close()
    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # ilk periyotlarda / hacim=0 durumunda notr deger


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]

    sma_20 = df["close"].rolling(20).mean()
    df["price_to_sma20"] = df["close"] / sma_20 - 1
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["rsi_14"] = _rsi(df["close"], 14)

    vol_mean_20 = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)

    # hedef: BIR SONRAKI mumun log-getirisi (bugunku satirdan gelecegi tahmin)
    df[TARGET_COLUMN] = df["log_return"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


def load_macro_daily() -> pd.DataFrame:
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {MACRO_TABLE} ORDER BY date").fetchdf()
    finally:
        con.close()
    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """BTC mumlarina (df) makro ozellikleri ekler.

    df, en az bir 'datetime' sutunu icermeli (load_ohlcv()'den gelen ham
    hali gibi). Donen df, MACRO_FEATURE_COLUMNS'daki sutunlarla genisletilmis
    olur; makro veri kapsamayan (cok eski/cok yeni) satirlarda bu sutunlar
    NaN kalir - cagiran taraf bunlari dropna ile temizlemeli.
    """
    macro = load_macro_daily().copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date")

    # Geleneksel piyasalar (S&P500, altin, brent, gumus, tahvil faizi)
    # hafta sonu/resmi tatil gunlerinde islem gormedigi icin o gunler
    # icin yfinance NaN doner (kripto sutunlari ise 7/24 islem gordugu
    # icin her zaman dolu). Getiriyi hesaplamadan ONCE bu sutunlari ileri
    # dolduruyoruz (ffill): boylece hafta sonu "son bilinen kapanis"
    # degeri tasinmis olur ve o gunlerin getirisi 0 (degisim yok) cikar.
    # Bunu yapmazsak, mesela Pazartesi'nin getirisi (bir onceki gun olan
    # Pazar NaN oldugu icin) de NaN olurdu - bu da her hafta bir gunun
    # (ve o gune denk gelen TUM 5dk'lik BTC mumlarinin) gereksiz yere
    # atilmasina yol acardi (~haftanin 1/7'si kadar veri kaybi).
    for col in _MACRO_RETURN_SOURCE_COLUMNS + _MACRO_DIFF_SOURCE_COLUMNS:
        macro[col] = macro[col].ffill()

    for col in _MACRO_RETURN_SOURCE_COLUMNS:
        macro[f"macro_{col}_ret"] = np.log(macro[col] / macro[col].shift(1))
    for col in _MACRO_DIFF_SOURCE_COLUMNS:
        macro[f"macro_{col}_chg"] = macro[col].diff()

    macro = macro[["date"] + MACRO_FEATURE_COLUMNS].dropna()

    # LEAKAGE ONLEMI: bir gunun BTC mumlari, o gunun DEGIL, bir onceki
    # gunun makro kapanisini/degisimini gormeli - bu yuzden makro
    # tarihini 1 gun ILERI kaydiriyoruz (14 Mart verisi artik 15 Mart
    # etiketiyle eslesecek). Merge anahtari olarak saf 'date' (saat/
    # timezone'suz) kullaniyoruz, yoksa tz-aware/tz-naive uyusmazligi
    # merge'i patlatir.
    macro["date"] = (macro["date"] + pd.Timedelta(days=1)).dt.date

    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)
    df["date"] = dt.dt.date
    df = df.merge(macro, on="date", how="left")
    return df.drop(columns=["date"])


def load_xaut_ohlcv() -> pd.DataFrame:
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {config.XAUT_TABLE} ORDER BY timestamp").fetchdf()
    finally:
        con.close()
    return df


def add_xaut_feature(df: pd.DataFrame) -> pd.DataFrame:
    """BTC mumlarina (df) XAUT'un (altin) ayni anki getirisini ekler.

    Makro tablosunun aksine burada 1 gunluk ileri kaydirma YAPILMAZ:
    XAUT de BTC gibi Binance'de 7/24 islem goruyor ve AYNI mum
    araliginda kapaniyor, yani df'deki bir satirin xaut_log_return'u
    ile log_return'u tam olarak ayni anda (o mumun kapanisinda)
    biliniyor - leakage riski yok, tipki BTC'nin kendi ozellikleri gibi.

    XAUT verisinin kapsamadigi (cok eski/yeni) satirlarda
    XAUT_FEATURE_COLUMNS NaN kalir - cagiran taraf dropna ile temizlemeli.
    """
    xaut = load_xaut_ohlcv().copy()
    xaut["xaut_log_return"] = np.log(xaut["close"] / xaut["close"].shift(1))
    xaut = xaut[["timestamp", "xaut_log_return"]].dropna()

    df = df.copy()
    df = df.merge(xaut, on="timestamp", how="left")
    return df


def time_based_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def build_sequences(features: np.ndarray, targets: np.ndarray, window: int):
    """Kaydirmali pencere ile (X, y) ciftleri olusturur.

    X[i] = features[i : i+window]   (gecmis `window` mum)
    y[i] = targets[i + window - 1]  (o pencerenin son gunundeki 'bir sonraki getiri' hedefi)
    """
    X, y = [], []
    for i in range(len(features) - window + 1):
        X.append(features[i : i + window])
        y.append(targets[i + window - 1])
    return np.array(X), np.array(y)


def prepare_dataset(
    window: int = None,
    include_macro: bool = False,
    include_xaut: bool = False,
    include_calendar: bool = False,
) -> DatasetSplit:
    window = window or config.WINDOW

    df = load_ohlcv()
    df = add_features(df)

    feature_columns = list(FEATURE_COLUMNS)
    if include_macro:
        df = add_macro_features(df)
        # makro verinin kapsamadigi (cok eski) satirlari at - bunlar
        # icin macro_* sutunlari NaN kalmisti
        df = df.dropna(subset=MACRO_FEATURE_COLUMNS).reset_index(drop=True)
        feature_columns = feature_columns + MACRO_FEATURE_COLUMNS

    if include_xaut:
        df = add_xaut_feature(df)
        df = df.dropna(subset=XAUT_FEATURE_COLUMNS).reset_index(drop=True)
        feature_columns = feature_columns + XAUT_FEATURE_COLUMNS

    if include_calendar:
        # add_calendar_features hicbir zaman NaN uretmez (bkz. modulun
        # kendisi), o yuzden dropna gerekmiyor.
        df = add_calendar_features(df)
        feature_columns = feature_columns + CALENDAR_FEATURE_COLUMNS

    train_df, val_df, test_df = time_based_split(df)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].values)

    def _scaled(d):
        return scaler.transform(d[feature_columns].values)

    X_train, y_train = build_sequences(_scaled(train_df), train_df[TARGET_COLUMN].values, window)
    X_val, y_val = build_sequences(_scaled(val_df), val_df[TARGET_COLUMN].values, window)
    X_test, y_test = build_sequences(_scaled(test_df), test_df[TARGET_COLUMN].values, window)

    # test setindeki her pencerenin son mumunun kapanis fiyati (fiyata donusum icin)
    close_test = test_df["close"].values[window - 1 :]

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        close_test=close_test,
        scaler=scaler,
    )
