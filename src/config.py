"""
Ortam degiskenlerini (.env) okuyup tek bir yerden proje genelinde
kullanilabilir hale getirir. Yeni bir alt-model (sentiment, hacim,
orderbook, karar agaci) eklendikce buraya yeni config degerleri
eklenebilir.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Proje kok dizini (bu dosyanin iki ust klasoru)
ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# --- Borsa / veri ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))

# --- Makro/cross-asset veri (CoinGecko + yfinance) ---
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
# True yaparsan LSTM, makro_daily tablosundaki ozellikleri de kullanir
# (once python -m src.data.fetch_macro_data calistirilmis olmali)
INCLUDE_MACRO = _get_bool("INCLUDE_MACRO", False)

# --- XAUT (Tether Gold, Binance'de 7/24 islem goren altin tokeni) ---
# Geleneksel altin verisi (yfinance) sadece piyasa saatlerinde guncelleniyor;
# XAUT ise BTC gibi Binance'de 7/24 islem goruyor ve spot altin fiyatini
# cok yakin takip ediyor (ort. sapma ~0.03$/ons). Bu yuzden BTC ile AYNI
# mum araliginda (5dk/1s/vb.) cekilip dogrudan zaman damgasina gore
# birlestiriliyor (bkz. dataset.py:add_xaut_feature) - gunluk makro veri
# gibi 1 gun kaydirmaya gerek yok, cunku ikisi de ayni anda kapanan
# mumlar.
XAUT_SYMBOL = os.getenv("XAUT_SYMBOL", "XAUT/USDT")
XAUT_TABLE = "xaut_ohlcv"
# True yaparsan LSTM, xaut_ohlcv tablosundaki XAUT getirisini de kullanir
# (once python -m src.data.fetch_xaut_data calistirilmis olmali)
INCLUDE_XAUT = _get_bool("INCLUDE_XAUT", False)

# --- LLM Agent (Katman 2) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Guncel model id'lerini https://docs.claude.com/en/docs/about-claude/models
# sayfasindan kontrol et - burada varsayilan olarak makul bir Sonnet
# modeli birakildi, istersen .env'de degistir.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

# --- Otomasyon (Katman 3: emir gonderme + Telegram) ---
# GUVENLIK: varsayilan olarak TRUE - gercek emir gonderilmez, sadece
# ne yapilacagi loglanir. Bilerek False yapmadan gercek parayla islem
# ACILMAZ (bkz. src/automation/order_execution.py).
DRY_RUN = _get_bool("DRY_RUN", True)

# True yaparsan (VE DRY_RUN=false ise) order_execution.py, GERCEK
# hesabina degil, Binance Futures Testnet'e (sahte parayla) emir
# gonderir - testnet.binancefuture.com'dan alinan AYRI bir API
# anahtari/secret gerekir (mainnet anahtarinla calismaz). Uctan uca
# akisi risksiz dogrulamak icin: DRY_RUN=false + BINANCE_TESTNET=true.
# Fiyat/orderbook verisi (LSTM, sinyal) HER ZAMAN gercek Binance'ten
# gelir - sadece emir gonderme/bakiye kismi testnet'e gider (testnet'in
# emir defteri gercekci degil, sinyal icin kullanilmiyor).
BINANCE_TESTNET = _get_bool("BINANCE_TESTNET", False)
TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "")
TESTNET_API_SECRET = os.getenv("TESTNET_API_SECRET", "")
# Her islemde toplam sermayenin ne kadari (kaldiractan ONCE) marjin
# olarak kullanilsin - ör. 0.1 = sermayenin %10'u
POSITION_SIZE_FRACTION = float(os.getenv("POSITION_SIZE_FRACTION", "0.1"))
# Gunluk oturum basi sermayeye gore, bu yuzdeyi asan bir kayipta
# sistem otomatik olarak durur (global zarar kurali / circuit breaker)
GLOBAL_LOSS_LIMIT_PCT = float(os.getenv("GLOBAL_LOSS_LIMIT_PCT", "5"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Ekonomik takvim (FOMC/CPI/NFP) ---
# Statik, elle derlenmis resmi takvim (bkz. src/data/economic_calendar.py)
# - orderbook'un aksine gecmise de uygulanabilir (leakage riski yok).
INCLUDE_CALENDAR = _get_bool("INCLUDE_CALENDAR", False)

# --- Orderbook (emir defteri) ---
# Binance sadece ANLIK orderbook veriyor (gecmise donuk cekilemez), bu
# yuzden bu, LSTM'in gecmis egitim verisine degil, LLM Agent'in canli
# karar anina ait bir sinyal (bkz. src/data/orderbook.py).
ORDERBOOK_DEPTH = int(os.getenv("ORDERBOOK_DEPTH", "20"))
ORDERBOOK_TABLE = "orderbook_snapshots"

# --- Model ---
WINDOW = int(os.getenv("WINDOW", "60"))

# --- Klasorler ---
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")
MODEL_DIR = ROOT_DIR / os.getenv("MODEL_DIR", "artifacts")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = DATA_DIR / "market.duckdb"
OHLCV_TABLE = "ohlcv_5m"
