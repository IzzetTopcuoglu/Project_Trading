"""
Uctan uca dongu: Katman 1 (LSTM) -> Katman 2 (LLM Agent) -> Katman 3
(Otomasyon: emir gonderme + Telegram).

ONEMLI: bunu calistirmadan once sirasiyla asagidakilerin tamamlanmis
olmasi gerekiyor (README'deki calistirma sirasina bak):
  1. python -m src.data.fetch_data                (BTC verisi)
  2. python -m src.models.lstm_direction.train     (LSTM modeli egitilmis olmali)
  3. .env'de ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID dolu olmali
  4. DRY_RUN=true ile once birkac tur calistirip ciktilari (Telegram
     mesajlari + konsol) incele; her seyi mantikli buluyorsan DRY_RUN=false yap

Kullanim:
    python -m src.main
"""
import time

import ccxt

from src import config
from src.agent.llm_agent import gather_context, decide
from src.automation import telegram_bot
from src.automation.order_execution import place_trade


def run_once() -> dict:
    context = gather_context()
    print(f"[main] baglam toplandi: {context}")

    decision = decide(context)
    print(f"[main] LLM Agent karari: {decision}")

    execution = place_trade(decision)
    print(f"[main] uygulama sonucu: {execution}")

    telegram_bot.send_status_message(decision, execution)
    telegram_bot.process_updates_once()

    return {"context": context, "decision": decision, "execution": execution}


def _seconds_until_next_candle() -> int:
    exchange = ccxt.binanceusdm()
    return max(exchange.parse_timeframe(config.TIMEFRAME), 5)


def main():
    print(f"[main] baslatildi - DRY_RUN={config.DRY_RUN}  TIMEFRAME={config.TIMEFRAME}  SYMBOL={config.SYMBOL}")
    if config.DRY_RUN:
        print("[main] DRY_RUN=true: gercek emir GONDERILMEYECEK, sadece plan loglanacak.")
    else:
        print("[main] DIKKAT: DRY_RUN=false - GERCEK EMIRLER Binance'e gonderilecek.")

    while True:
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001 - dongu tek bir hatada tamamen durmasin
            print(f"[main] HATA (bu turu atlıyorum): {exc!r}")
        time.sleep(_seconds_until_next_candle())


if __name__ == "__main__":
    main()
