"""
Katman 3'un emir gonderme kismini (order_execution.place_trade) GERCEK
Binance Futures Testnet API'sine karsi, uctan uca dogrulamak icin tek
seferlik bir script. tests_automation.py'daki testlerden farki: o
dosya hicbir gercek API cagrisi yapmaz (hepsi mock), bu script ise
gercekten testnet.binancefuture.com'a baglanir.

ONKOSUL - .env dosyanda (ORIJINAL .env, .env.example degil):
    DRY_RUN=false
    BINANCE_TESTNET=true
    TESTNET_API_KEY=<testnet'ten aldigin key>
    TESTNET_API_SECRET=<testnet'ten aldigin secret>

Ne yapar:
    1. Testnet hesap bakiyeni ceker ve ekrana yazar (baglanti/anahtar
       dogru mu kontrolu).
    2. LLM Agent'i ATLAYIP elle kucuk, sabit bir "long" karari olusturur
       (boylece bu script API kredisi harcamaz, sadece emir gonderme
       mekanigini test eder).
    3. place_trade() cagirir - bu, gercekten testnet'te bir market
       giris emri + STOP_MARKET korumali emir acar (sahte parayla).
    4. Sonucu (entry_order/stop_order id'leri ya da hata mesaji)
       ekrana yazar.

Calistir:
    python run_testnet_order_test.py

NOT: config.py .env'i modul import edilirken bir kere okur, bu yuzden
.env'i degistirdikten sonra (varsa) eski bir Python sürecini kapatip
bu scripti yeniden calistir.
"""
from src import config
from src.automation.order_execution import get_account_equity, place_trade

# Kasitli olarak kucuk ve sabit - gercek bir LLM/LSTM karari degil,
# sadece emir gonderme mekanigini (leverage ayarlama + market giris +
# stop-limit korumasi) dogrulamak icin.
TEST_DECISION = {
    "direction": "long",
    "confidence": 0.9,
    "leverage": 1,
    "stop_limit_pct": 1.0,
    "signals_aligned": True,
    "rationale": "run_testnet_order_test.py ile elle tetiklenen uctan uca testnet dogrulamasi",
}


def main():
    print(f"DRY_RUN={config.DRY_RUN}  BINANCE_TESTNET={config.BINANCE_TESTNET}  SYMBOL={config.SYMBOL}")

    if config.DRY_RUN:
        print("\nHATA: DRY_RUN=true iken bu script gercek testnet emri gonderemez.")
        print(".env dosyanda DRY_RUN=false yap ve tekrar dene.")
        return

    if not config.BINANCE_TESTNET:
        print("\nUYARI: BINANCE_TESTNET=false! Bu sekilde devam edersen GERCEK hesabina")
        print("emir gitmeye calisir. Once .env'de BINANCE_TESTNET=true yap.")
        return

    if not config.TESTNET_API_KEY or not config.TESTNET_API_SECRET:
        print("\nHATA: TESTNET_API_KEY / TESTNET_API_SECRET .env'de bos.")
        return

    print("\n1) Testnet hesap bakiyesi cekiliyor...")
    try:
        equity = get_account_equity()
    except Exception as exc:  # noqa: BLE001
        print(f"HATA: bakiye cekilemedi -> {exc!r}")
        print("Kontrol et: API key/secret dogru mu, testnet hesabinda 'Enable Futures' aktif mi?")
        return
    print(f"   Testnet USDT bakiyesi: {equity}")

    print("\n2) Test karari gonderiliyor:", TEST_DECISION)
    result = place_trade(TEST_DECISION, current_equity=equity)

    print("\n3) Sonuc:")
    print(result)

    if result["status"] == "executed":
        print("\nBASARILI - testnet.binancefuture.com uzerindeki Futures hesabindan")
        print("pozisyonu ve emirleri gorsel olarak da dogrulayabilirsin.")
    else:
        print(f"\nEmir GONDERILMEDI (status={result['status']}) - yukaridaki 'reason' alanina bak.")


if __name__ == "__main__":
    main()
