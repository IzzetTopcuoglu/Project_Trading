"""
risk_state / order_execution / telegram_bot icin test dosyasi.
Hicbir gercek Binance/Telegram API cagrisi yapmaz (hepsi mock/gecici
dosya). Proje kokunde calistir:
    python tests_automation.py
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.automation import risk_state, telegram_bot
from src.automation.order_execution import place_trade, _make_exchange


def _tmp_state_path():
    return Path(tempfile.mkdtemp()) / "risk_state.json"


# ---------- risk_state ----------

def test_ensure_session_creates_and_keeps_same_day():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p):
        s1 = risk_state.ensure_session(1000.0)
        s2 = risk_state.ensure_session(1000.0)  # ayni gun -> degismemeli
        assert s1["session_start_equity"] == 1000.0
        assert s2["session_start_equity"] == 1000.0
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_ensure_session_creates_and_keeps_same_day OK")


def test_check_global_loss_limit_not_breached():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p), patch("src.config.GLOBAL_LOSS_LIMIT_PCT", 5.0):
        risk_state.ensure_session(1000.0)
        breached = risk_state.check_global_loss_limit(980.0)  # %2 kayip, limit %5
        assert breached is False
        assert risk_state.is_stopped() is False
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_check_global_loss_limit_not_breached OK")


def test_check_global_loss_limit_breached_sets_stopped():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p), patch("src.config.GLOBAL_LOSS_LIMIT_PCT", 5.0):
        risk_state.ensure_session(1000.0)
        breached = risk_state.check_global_loss_limit(900.0)  # %10 kayip, limit %5
        assert breached is True
        assert risk_state.is_stopped() is True
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_check_global_loss_limit_breached_sets_stopped OK")


def test_manual_set_stopped_and_resume():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p):
        risk_state.set_stopped(True, reason="test")
        assert risk_state.is_stopped() is True
        risk_state.set_stopped(False)
        assert risk_state.is_stopped() is False
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_manual_set_stopped_and_resume OK")


# ---------- order_execution.place_trade ----------

NO_TRADE_DECISION = {"direction": "no_trade", "confidence": 0.4, "leverage": 1, "stop_limit_pct": 1.0, "signals_aligned": False, "rationale": "belirsiz"}
LONG_DECISION = {"direction": "long", "confidence": 0.8, "leverage": 5, "stop_limit_pct": 1.0, "signals_aligned": True, "rationale": "uyumlu"}


def test_place_trade_no_trade():
    result = place_trade(NO_TRADE_DECISION)
    assert result["status"] == "no_trade"
    print("test_place_trade_no_trade OK")


def test_place_trade_blocked_when_manually_stopped():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p):
        risk_state.set_stopped(True, reason="manuel test durdurmasi")
        result = place_trade(LONG_DECISION, current_equity=1000.0)
        assert result["status"] == "stopped", result
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_place_trade_blocked_when_manually_stopped OK")


def test_place_trade_dry_run_returns_plan():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p), \
         patch("src.config.DRY_RUN", True), \
         patch("src.config.POSITION_SIZE_FRACTION", 0.1), \
         patch("src.config.GLOBAL_LOSS_LIMIT_PCT", 5.0):
        result = place_trade(LONG_DECISION, current_equity=1000.0)
    assert result["status"] == "dry_run", result
    assert result["plan"]["position_value_usdt"] == 500.0  # 1000 * 0.1 * 5x kaldirac
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_place_trade_dry_run_returns_plan OK ->", result["plan"])


def test_place_trade_blocked_by_global_loss_limit():
    p = _tmp_state_path()
    with patch("src.automation.risk_state.STATE_PATH", p), \
         patch("src.config.DRY_RUN", True), \
         patch("src.config.GLOBAL_LOSS_LIMIT_PCT", 5.0):
        risk_state.ensure_session(1000.0)
        result = place_trade(LONG_DECISION, current_equity=900.0)  # %10 kayip
    assert result["status"] == "stopped", result
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_place_trade_blocked_by_global_loss_limit OK ->", result["reason"])


# ---------- order_execution._make_exchange (testnet secimi) ----------

def test_make_exchange_uses_mainnet_by_default():
    with patch("src.config.BINANCE_TESTNET", False), \
         patch("src.config.BINANCE_API_KEY", "mainnet-key"), \
         patch("src.config.TESTNET_API_KEY", "testnet-key"):
        exchange = _make_exchange()
    assert exchange.apiKey == "mainnet-key"
    assert "testnet.binancefuture.com" not in str(exchange.urls.get("api"))
    print("test_make_exchange_uses_mainnet_by_default OK")


def test_make_exchange_switches_to_testnet():
    with patch("src.config.BINANCE_TESTNET", True), \
         patch("src.config.TESTNET_API_KEY", "testnet-key"), \
         patch("src.config.TESTNET_API_SECRET", "testnet-secret"):
        exchange = _make_exchange()
    assert exchange.apiKey == "testnet-key"
    assert "testnet.binancefuture.com" in str(exchange.urls.get("api"))
    print("test_make_exchange_switches_to_testnet OK ->", exchange.urls["api"]["fapiPrivate"])


def test_make_exchange_testnet_suppresses_ccxt_sandbox_warning():
    # ccxt 4.x, binanceusdm'de set_sandbox_mode(True) sonrasi gercek bir
    # API cagrisi denendiginde "testnet/sandbox mode is not supported for
    # futures anymore" diye NotSupported firlatiyor (kendi ic guvenlik
    # onlemi). testnet.binancefuture.com fiilen calistigi icin bu ozelligi
    # bilerek bastiriyoruz - bu test o bayragin unutulmadigini dogrular.
    with patch("src.config.BINANCE_TESTNET", True), \
         patch("src.config.TESTNET_API_KEY", "testnet-key"), \
         patch("src.config.TESTNET_API_SECRET", "testnet-secret"):
        exchange = _make_exchange()
    assert exchange.options.get("disableFuturesSandboxWarning") is True
    print("test_make_exchange_testnet_suppresses_ccxt_sandbox_warning OK")


# ---------- telegram_bot ----------

def test_format_decision_message():
    msg = telegram_bot.format_decision_message(LONG_DECISION | {"timestamp": "2026-01-01T00:00:00Z"}, {"status": "dry_run"})
    assert "long" in msg and "dry_run" in msg
    print("test_format_decision_message OK")


def test_send_message_without_config_is_skipped():
    with patch("src.config.TELEGRAM_BOT_TOKEN", ""), patch("src.config.TELEGRAM_CHAT_ID", ""):
        result = telegram_bot.send_message("test mesaji")
    assert result["status"] == "skipped_no_config"
    print("test_send_message_without_config_is_skipped OK")


def test_handle_command_durdur_and_devam():
    p = _tmp_state_path()
    fake_post = MagicMock(return_value=MagicMock(json=lambda: {"ok": True}, raise_for_status=lambda: None))
    with patch("src.automation.risk_state.STATE_PATH", p), \
         patch("src.config.TELEGRAM_BOT_TOKEN", "fake-token"), \
         patch("src.config.TELEGRAM_CHAT_ID", "123"), \
         patch("src.automation.telegram_bot.requests.post", fake_post):
        telegram_bot.handle_command("/durdur", chat_id="123")
        assert risk_state.is_stopped() is True

        telegram_bot.handle_command("/devam", chat_id="123")
        assert risk_state.is_stopped() is False
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_handle_command_durdur_and_devam OK -> gonderilen mesaj sayisi:", fake_post.call_count)


def test_process_updates_once_dispatches_command():
    p = _tmp_state_path()
    offset_path = p.parent / "telegram_offset.json"
    fake_updates = [{"update_id": 42, "message": {"text": "/durdur", "chat": {"id": 555}}}]

    fake_get = MagicMock(return_value=MagicMock(json=lambda: {"result": fake_updates}, raise_for_status=lambda: None))
    fake_post = MagicMock(return_value=MagicMock(json=lambda: {"ok": True}, raise_for_status=lambda: None))

    with patch("src.automation.risk_state.STATE_PATH", p), \
         patch("src.automation.telegram_bot.OFFSET_PATH", offset_path), \
         patch("src.config.TELEGRAM_BOT_TOKEN", "fake-token"), \
         patch("src.config.TELEGRAM_CHAT_ID", "123"), \
         patch("src.automation.telegram_bot.requests.get", fake_get), \
         patch("src.automation.telegram_bot.requests.post", fake_post):
        n = telegram_bot.process_updates_once()
        assert n == 1
        assert risk_state.is_stopped() is True
    shutil.rmtree(p.parent, ignore_errors=True)
    print("test_process_updates_once_dispatches_command OK")


if __name__ == "__main__":
    test_ensure_session_creates_and_keeps_same_day()
    test_check_global_loss_limit_not_breached()
    test_check_global_loss_limit_breached_sets_stopped()
    test_manual_set_stopped_and_resume()
    test_place_trade_no_trade()
    test_place_trade_blocked_when_manually_stopped()
    test_place_trade_dry_run_returns_plan()
    test_place_trade_blocked_by_global_loss_limit()
    test_make_exchange_uses_mainnet_by_default()
    test_make_exchange_switches_to_testnet()
    test_make_exchange_testnet_suppresses_ccxt_sandbox_warning()
    test_format_decision_message()
    test_send_message_without_config_is_skipped()
    test_handle_command_durdur_and_devam()
    test_process_updates_once_dispatches_command()
    print("\nTUM TESTLER GECTI")
