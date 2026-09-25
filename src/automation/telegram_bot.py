"""
Katman 3'un Telegram kismi: agent'in kararlarini/islem sonuclarini
anlik bildirir, ve /durdur - /devam - /durum komutlarini isler.

ONEMLI: bu, HER ISLEMI onaylatan bir kapi DEGIL - sadece izleme +
manuel acil durdurma icin. Agent kendi basina islem acar (bkz.
order_execution.py); Telegram sadece durumu bildirir ve /durdur ile
sistemi durdurma imkani sunar.

Kurulum: Telegram'da @BotFather'a yazip /newbot ile bir token al,
botuna bir mesaj atip sonra
https://api.telegram.org/bot<TOKEN>/getUpdates adresini tarayicida
acarak kendi chat_id'ni ogren. Ikisini de .env'e yaz (bkz. .env.example).

Kullanim (komutlari dinlemek icin bagimsiz test):
    python -m src.automation.telegram_bot
"""
import json
import time

import requests

from src import config
from src.automation import risk_state

OFFSET_PATH = config.DATA_DIR / "telegram_offset.json"


def _api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"


def send_message(text: str, chat_id: str = None) -> dict:
    chat_id = chat_id or config.TELEGRAM_CHAT_ID
    if not config.TELEGRAM_BOT_TOKEN or not chat_id:
        print(f"[telegram] UYARI: TELEGRAM_BOT_TOKEN/CHAT_ID tanimli degil - mesaj gonderilmedi:\n{text}")
        return {"status": "skipped_no_config"}
    resp = requests.post(_api_url("sendMessage"), data={"chat_id": chat_id, "text": text}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def format_decision_message(decision: dict, execution: dict) -> str:
    lines = [
        f"BTC Karar Destek Sistemi ({decision.get('timestamp', '')})",
        f"Yon: {decision.get('direction')}  |  Kaldirac: {decision.get('leverage')}x  |  Guven: {decision.get('confidence')}",
        f"Stop-limit: %{decision.get('stop_limit_pct')}",
        f"Gerekce: {decision.get('rationale')}",
        f"Uygulama durumu: {execution.get('status')}",
    ]
    if execution.get("status") in ("stopped", "error"):
        lines.append(f"Detay: {execution.get('reason')}")
    return "\n".join(lines)


def send_status_message(decision: dict, execution: dict, chat_id: str = None) -> dict:
    return send_message(format_decision_message(decision, execution), chat_id=chat_id)


def _load_offset() -> int:
    if OFFSET_PATH.exists():
        with open(OFFSET_PATH) as f:
            return json.load(f).get("last_update_id")
    return None


def _save_offset(update_id: int) -> None:
    OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OFFSET_PATH, "w") as f:
        json.dump({"last_update_id": update_id}, f)


def get_updates(offset: int = None, timeout: int = 0) -> list:
    if not config.TELEGRAM_BOT_TOKEN:
        return []
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(_api_url("getUpdates"), params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json().get("result", [])


def handle_command(text: str, chat_id) -> None:
    command = (text or "").strip().lower()
    if command == "/durdur":
        risk_state.set_stopped(True, reason="Telegram /durdur komutuyla manuel durduruldu")
        send_message("Sistem durduruldu - yeni islem acilmayacak. Tekrar baslatmak icin /devam yaz.", chat_id=chat_id)
    elif command == "/devam":
        risk_state.set_stopped(False)
        send_message("Sistem tekrar aktif - yeni kararlar otomatik uygulanacak.", chat_id=chat_id)
    elif command == "/durum":
        state = risk_state.load_state()
        durum = "DURDURULDU" if state.get("stopped") else "AKTIF"
        send_message(
            f"Durum: {durum}\n"
            f"Oturum baslangic sermayesi: {state.get('session_start_equity')}\n"
            f"Durdurulma nedeni (varsa): {state.get('stopped_reason')}",
            chat_id=chat_id,
        )
    else:
        send_message("Bilinen komutlar: /durum, /durdur, /devam", chat_id=chat_id)


def process_updates_once() -> int:
    """Bekleyen Telegram mesajlarini bir kez isler. Islenen komut sayisini doner."""
    last_offset = _load_offset()
    updates = get_updates(offset=(last_offset + 1) if last_offset is not None else None)

    n = 0
    for update in updates:
        _save_offset(update["update_id"])
        message = update.get("message") or {}
        text = message.get("text")
        chat_id = (message.get("chat") or {}).get("id")
        if text and chat_id:
            handle_command(text, chat_id)
            n += 1
    return n


def main():
    print("[telegram] komut dinleme basladi (Ctrl+C ile durdur)...")
    while True:
        n = process_updates_once()
        if n:
            print(f"[telegram] {n} komut islendi.")
        time.sleep(3)


if __name__ == "__main__":
    main()
