"""
Basit, dosya tabanli risk/durum takibi. Iki ayri "durdurma" kaynagini
TEK bir bayrakta (stopped) birlestirir:
  1. Telegram uzerinden manuel /durdur komutu
  2. Global zarar kurali (gunluk oturum basi sermayeye gore) otomatik tetiklenmesi

order_execution.place_trade() her islemden once is_stopped()'a bakar -
ikisinden hangisi tetiklenmis olursa olsun yeni islem acilmaz.
"""
import json
from datetime import date, timezone, datetime

from src import config

STATE_PATH = config.DATA_DIR / "risk_state.json"

DEFAULT_STATE = {
    "session_date": None,
    "session_start_equity": None,
    "stopped": False,
    "stopped_reason": None,
    "stopped_at": None,
}


def load_state() -> dict:
    if not STATE_PATH.exists():
        return dict(DEFAULT_STATE)
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def ensure_session(current_equity: float) -> dict:
    """Gun degistiyse (ya da hic oturum yoksa) yeni bir oturum baslatir -
    o gunun basindaki sermayeyi baz alir. Ayni gun icinde tekrar
    cagrilirsa mevcut oturumu degistirmez."""
    state = load_state()
    today = date.today().isoformat()
    if state.get("session_date") != today:
        state = dict(DEFAULT_STATE)
        state["session_date"] = today
        state["session_start_equity"] = current_equity
        save_state(state)
    return state


def is_stopped() -> bool:
    return bool(load_state().get("stopped", False))


def set_stopped(stopped: bool, reason: str = None) -> None:
    state = load_state()
    state["stopped"] = stopped
    state["stopped_reason"] = reason
    state["stopped_at"] = datetime.now(timezone.utc).isoformat() if stopped else None
    save_state(state)


def check_global_loss_limit(current_equity: float, limit_pct: float = None) -> bool:
    """Oturum basindaki sermayeye gore yuzde kac kaybedildigini kontrol
    eder. Limit asilmissa OTOMATIK OLARAK set_stopped(True) yapar ve
    True doner (cagiran taraf ayrica bir sey yapmasa da sistem durmus olur)."""
    limit_pct = config.GLOBAL_LOSS_LIMIT_PCT if limit_pct is None else limit_pct
    state = ensure_session(current_equity)
    start_equity = state.get("session_start_equity") or current_equity

    if start_equity <= 0:
        return False

    drawdown_pct = (current_equity - start_equity) / start_equity * 100
    breached = drawdown_pct <= -abs(limit_pct)

    if breached and not state.get("stopped"):
        set_stopped(
            True,
            reason=(
                f"Global zarar kurali tetiklendi: oturum ici degisim {drawdown_pct:.2f}% "
                f"(limit: -{abs(limit_pct)}%)"
            ),
        )
    return breached
