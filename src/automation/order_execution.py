"""
Katman 3'un emir gonderme kismi: LLM Agent'in kararini (bkz.
src/agent/llm_agent.py) ccxt uzerinden Binance Futures'a stop-limit
korumali bir pozisyon olarak gonderir.

GUVENLIK TASARIMI (onemli) - UC KADEMELI test/calisma modu var:
  1. DRY_RUN=true (varsayilan): gercek emir HICBIR YERE gonderilmez,
     sadece ne yapilacagi loglanir/donulur.
  2. DRY_RUN=false + BINANCE_TESTNET=true: gercek emir gonderilir AMA
     Binance Futures TESTNET'ine (testnet.binancefuture.com, sahte
     parayla, AYRI bir API anahtariyla) - uctan uca akisi (leverage
     ayarlama, emir olusturma, stop emri) gercek borsa API'siyle,
     risksiz dogrulamak icin. bkz. _make_exchange().
  3. DRY_RUN=false + BINANCE_TESTNET=false: GERCEK hesabina GERCEK
     parayla emir gider. Once mutlaka 1 ve 2'yi tamamla.
  - Her cagrida once risk_state.is_stopped() kontrol edilir - Telegram'dan
    /durdur ile ya da global zarar kuraliyla durdurulmus bir sistem
    yeni islem ACMAZ.
  - decision["direction"] == "no_trade" ise zaten hicbir sey yapilmaz.

NOT (dogrulandi, 2026-09-20): Binance Futures'da stop emri icin
"STOP_MARKET" + reduceOnly=True kombinasyonu, testnet'te gercek bir
hesapla basariyla test edildi (bkz. run_testnet_order_test.py) - hem
market giris emri hem de STOP_MARKET/reduceOnly korumali emir dogru
sekilde olusturuldu ve Binance tarafinda kabul edildi (protective stop,
Binance'de klasik "order" degil "algo order"/algoId olarak donuyor,
ccxt bunu unified create_order() uzerinden sorunsuz yonetiyor).

NOT 2: ccxt 4.x, binanceusdm icin sandbox modunu "artik resmi olarak
desteklemiyoruz" diyerek set_sandbox_mode(True) sonrasi bir NotSupported
hatasi firlatiyor (bkz. _make_exchange() icindeki disableFuturesSandboxWarning
notu) - bu, testnet.binancefuture.com'un kapandigi anlamina gelmiyor,
sadece ccxt'nin kendi bakim/garanti kapsamindan cikardigi anlamina
geliyor. Bu bayrak sadece o hatayi bastiriyor, URL yonlendirmesi zaten
dogru calisiyor.
"""
import ccxt

from src import config
from src.automation import risk_state


def _make_exchange() -> ccxt.Exchange:
    """Emir gonderme/bakiye icin kullanilan borsa baglantisi.

    BINANCE_TESTNET=true ise Binance Futures Testnet'e (sahte parayla,
    AYRI bir API anahtariyla) baglanir - gercek hesaba HICBIR sekilde
    dokunmaz. Fiyat/orderbook verisi (src/data/fetch_data.py,
    src/data/orderbook.py) bundan etkilenmez, onlar her zaman gercek
    Binance'i kullanmaya devam eder (testnet'in emir defteri/hacmi
    gercekci olmadigi icin sinyal olarak kullanilmaz)."""
    if config.BINANCE_TESTNET:
        exchange = ccxt.binanceusdm(
            {
                "apiKey": config.TESTNET_API_KEY or None,
                "secret": config.TESTNET_API_SECRET or None,
                "enableRateLimit": True,
            }
        )
        exchange.set_sandbox_mode(True)
        # ccxt, Binance Futures testnet'i "artik resmi olarak desteklemiyoruz"
        # diyerek varsayilan olarak burada bir NotSupported hatasi firlatiyor
        # (bkz. https://t.me/ccxt_announcements/92). Ancak testnet.binancefuture.com
        # fiilen hala calisan/erisilebilir bir servis - ccxt sadece bunu aktif
        # bakim/garanti kapsamindan cikardigini belirtiyor. set_sandbox_mode()
        # zaten URL'leri dogru sekilde testnet.binancefuture.com'a yonlendirdigi
        # icin (yukarida dogrulandi), bu bayrak sadece o kendinden-savunma
        # uyarisini/hatasini bastirip gercek API cagrisinin yapilmasina izin
        # veriyor - bir guvenlik acigi degil, ccxt'nin ic kontrolu.
        exchange.options["disableFuturesSandboxWarning"] = True
        return exchange

    return ccxt.binanceusdm(
        {
            "apiKey": config.BINANCE_API_KEY or None,
            "secret": config.BINANCE_API_SECRET or None,
            "enableRateLimit": True,
        }
    )


def get_account_equity(exchange: ccxt.Exchange = None) -> float:
    exchange = exchange or _make_exchange()
    balance = exchange.fetch_balance()
    return float(balance["total"]["USDT"])


def place_trade(decision: dict, symbol: str = None, current_equity: float = None, exchange: ccxt.Exchange = None) -> dict:
    """decision, src/agent/schema.py:DECISION_SCHEMA'ya uyan bir dict
    olmali (bkz. llm_agent.decide()). Sonuc her zaman bir "status"
    alani icerir: no_trade | stopped | dry_run | executed | error."""
    symbol = symbol or config.SYMBOL

    if decision.get("direction") == "no_trade":
        return {"status": "no_trade", "reason": decision.get("rationale")}

    if risk_state.is_stopped():
        state = risk_state.load_state()
        return {
            "status": "stopped",
            "reason": state.get("stopped_reason") or "Sistem durduruldu (/durdur ya da global zarar kurali)",
        }

    try:
        if current_equity is None:
            current_equity = get_account_equity(exchange)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"Hesap bakiyesi alinamadi: {exc!r}"}

    if risk_state.check_global_loss_limit(current_equity):
        state = risk_state.load_state()
        return {"status": "stopped", "reason": state["stopped_reason"]}

    side = "buy" if decision["direction"] == "long" else "sell"
    leverage = decision["leverage"]
    position_value = current_equity * config.POSITION_SIZE_FRACTION * leverage

    order_plan = {
        "symbol": symbol,
        "side": side,
        "leverage": leverage,
        "position_value_usdt": round(position_value, 2),
        "stop_limit_pct": decision["stop_limit_pct"],
    }

    if config.DRY_RUN:
        print(f"[order] DRY_RUN aktif - gercek emir GONDERILMEDI. Plan: {order_plan}")
        return {"status": "dry_run", "plan": order_plan}

    try:
        exchange = exchange or _make_exchange()
        exchange.set_leverage(leverage, symbol)

        ticker = exchange.fetch_ticker(symbol)
        entry_price = ticker["last"]
        amount = position_value / entry_price

        entry_order = exchange.create_order(symbol, type="market", side=side, amount=amount)

        stop_side = "sell" if side == "buy" else "buy"
        stop_distance = entry_price * decision["stop_limit_pct"] / 100
        stop_price = entry_price - stop_distance if side == "buy" else entry_price + stop_distance

        stop_order = exchange.create_order(
            symbol,
            type="STOP_MARKET",
            side=stop_side,
            amount=amount,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )

        return {"status": "executed", "entry_order": entry_order, "stop_order": stop_order, "plan": order_plan}

    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"Emir gonderilirken hata: {exc!r}", "plan": order_plan}
