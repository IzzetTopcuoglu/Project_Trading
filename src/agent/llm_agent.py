"""
Katman 2 - LLM Agent. LSTM + orderbook + ekonomik takvim sinyallerini
bir araya getirip, Anthropic API'sine "tool use" (zorunlu fonksiyon
cagirma) ile yapilandirilmis, semaya uymaya ZORLANMIS bir JSON karar
uretir (bkz. schema.py). Karar Katman 3'e (otomasyon) dogrudan gider -
insan onayi yok, bu yuzden HATA DURUMUNDA ASLA rastgele/belirsiz bir
seye devam etmez, guvenli varsayilana (islem acma) duser.

Kullanim (tek seferlik canli test):
    python -m src.agent.llm_agent
"""
from datetime import datetime, timezone

from src import config
from src.agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from src.agent.schema import DECISION_TOOL, TOOL_NAME, validate_decision

# API/semadan gelen HERHANGI bir sorunda (eksik key, ag hatasi, semaya
# uymayan cikti, timeout, vb.) donulen guvenli varsayilan - agent
# otonom calistigi icin "bilmiyorum" durumunda ASLA rastgele bir yon
# uydurmaz, islem acmaz.
SAFE_DEFAULT_DECISION = {
    "direction": "no_trade",
    "confidence": 0.0,
    "leverage": 1,
    "stop_limit_pct": 1.0,
    "signals_aligned": False,
    "rationale": "LLM Agent karari alinamadi (API/sema hatasi) - guvenli varsayilan: islem acilmadi.",
}


def gather_context(symbol: str = None) -> dict:
    """Canli kullanim icin: LSTM + orderbook + takvimden anlik baglami toplar.

    NOT: bunu cagirmak icin egitilmis lstm_direction modeli (artifacts/
    altinda) ve gecerli bir Binance baglantisi gerekir - test ortaminda
    kullanilmaz, testler decide()'i hazir bir context dict ile cagirir.
    """
    from src.data.economic_calendar import get_calendar_context
    from src.data.orderbook import get_orderbook_context
    from src.models.lstm_direction.predict import predict_next_direction

    symbol = symbol or config.SYMBOL
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lstm": predict_next_direction(),
        "orderbook": get_orderbook_context(symbol=symbol),
        "calendar": get_calendar_context(),
    }


def _extract_tool_input(response) -> dict:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            return block.input
    raise ValueError("Yanitta beklenen tool_use blogu bulunamadi")


def decide(context: dict, model: str = None) -> dict:
    """Verilen baglamdan (context) yapilandirilmis bir karar uretir.

    Basarili olursa DECISION_SCHEMA'ya uyan bir dict + prompt_version/
    timestamp/context (denetim/log icin) doner. Herhangi bir hatada
    (API anahtari yok, ag hatasi, semaya uymayan cikti, vb.) SAFE_DEFAULT_DECISION
    dondurur - asla exception firlatmaz (otonom calisan bir sistemde
    ust katmani (otomasyon) crash etmemesi kritik).
    """
    result = dict(SAFE_DEFAULT_DECISION)
    result["prompt_version"] = PROMPT_VERSION
    result["timestamp"] = context.get("timestamp") or datetime.now(timezone.utc).isoformat()

    if not config.ANTHROPIC_API_KEY:
        print("[llm-agent] UYARI: ANTHROPIC_API_KEY tanimli degil - guvenli varsayilana dusuluyor.")
        return result

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model or config.LLM_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[DECISION_TOOL],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": build_user_message(context)}],
        )
        decision = _extract_tool_input(response)
        validate_decision(decision)

        decision["prompt_version"] = PROMPT_VERSION
        decision["timestamp"] = result["timestamp"]
        return decision

    except Exception as exc:  # noqa: BLE001 - kasitli genis yakalama, otonom sistem asla crash etmemeli
        print(f"[llm-agent] HATA: {exc!r} - guvenli varsayilana dusuluyor.")
        return result


if __name__ == "__main__":
    ctx = gather_context()
    print("[llm-agent] baglam:", ctx)
    print("[llm-agent] karar:", decide(ctx))
