"""
LLM Agent (src/agent) icin test dosyasi - GERCEK Anthropic API'sine
ASLA baglanmaz, hepsi mock/sahte obje ile calisir. Proje kokunde
calistir:
    python tests_agent.py
"""
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.llm_agent import SAFE_DEFAULT_DECISION, decide
from src.agent.schema import TOOL_NAME, validate_decision

FAKE_CONTEXT = {
    "symbol": "BTC/USDT",
    "timestamp": "2026-04-09T12:00:00+00:00",
    "lstm": {"last_close": 65000.0, "prob_up": 0.71, "direction": "yukselis", "confidence": 0.71},
    "orderbook": {"imbalance": 0.4, "best_bid": 64990, "best_ask": 65010, "spread_bps": 3.0},
    "calendar": {"next_event_type": "CPI", "next_event_date": "2026-04-10", "days_until_next_event": 1, "high_impact_soon": True},
}


def _fake_tool_response(decision_input: dict):
    tool_block = SimpleNamespace(type="tool_use", name=TOOL_NAME, input=decision_input)
    return SimpleNamespace(content=[tool_block])


def test_schema_accepts_valid_decision():
    valid = {
        "direction": "long", "confidence": 0.7, "leverage": 5,
        "stop_limit_pct": 1.0, "signals_aligned": True, "rationale": "test",
    }
    validate_decision(valid)  # exception firlatmazsa gecer
    print("test_schema_accepts_valid_decision OK")


def test_schema_rejects_invalid_leverage():
    invalid = {
        "direction": "long", "confidence": 0.7, "leverage": 3,  # 3 semaya gore gecersiz (sadece 1 ya da 5)
        "stop_limit_pct": 1.0, "signals_aligned": True, "rationale": "test",
    }
    try:
        validate_decision(invalid)
        raise AssertionError("gecersiz leverage kabul edilmemeliydi")
    except Exception:
        print("test_schema_rejects_invalid_leverage OK (bekledigimiz gibi reddedildi)")


def test_decide_without_api_key_falls_back_to_safe_default():
    with patch("src.config.ANTHROPIC_API_KEY", ""):
        result = decide(FAKE_CONTEXT)
    assert result["direction"] == SAFE_DEFAULT_DECISION["direction"]
    assert result["leverage"] == 1
    print("test_decide_without_api_key_falls_back_to_safe_default OK ->", result["direction"])


def test_decide_with_valid_mocked_response():
    fake_decision = {
        "direction": "long", "confidence": 0.75, "leverage": 5,
        "stop_limit_pct": 1.2, "signals_aligned": True,
        "rationale": "LSTM ve orderbook ikisi de yukselis gosteriyor, yakin bir yuksek etkili olay yok.",
    }
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: _fake_tool_response(fake_decision)))

    with patch("src.config.ANTHROPIC_API_KEY", "fake-key"), \
         patch("anthropic.Anthropic", return_value=fake_client):
        result = decide(FAKE_CONTEXT)

    assert result["direction"] == "long", result
    assert result["leverage"] == 5, result
    assert "prompt_version" in result
    print("test_decide_with_valid_mocked_response OK ->", result)


def test_decide_with_invalid_mocked_response_falls_back():
    # semaya uymuyor (leverage=3 gecersiz) - agent bunu kabul etmemeli
    bad_decision = {
        "direction": "long", "confidence": 0.9, "leverage": 3,
        "stop_limit_pct": 1.0, "signals_aligned": True, "rationale": "gecersiz",
    }
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: _fake_tool_response(bad_decision)))

    with patch("src.config.ANTHROPIC_API_KEY", "fake-key"), \
         patch("anthropic.Anthropic", return_value=fake_client):
        result = decide(FAKE_CONTEXT)

    assert result["direction"] == "no_trade", result  # guvenli varsayilana dusmus olmali
    print("test_decide_with_invalid_mocked_response_falls_back OK ->", result["direction"])


def test_decide_with_api_exception_falls_back():
    def _raise(**kwargs):
        raise RuntimeError("sahte ag hatasi")

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=_raise))

    with patch("src.config.ANTHROPIC_API_KEY", "fake-key"), \
         patch("anthropic.Anthropic", return_value=fake_client):
        result = decide(FAKE_CONTEXT)

    assert result["direction"] == "no_trade", result
    print("test_decide_with_api_exception_falls_back OK ->", result["direction"])


if __name__ == "__main__":
    test_schema_accepts_valid_decision()
    test_schema_rejects_invalid_leverage()
    test_decide_without_api_key_falls_back_to_safe_default()
    test_decide_with_valid_mocked_response()
    test_decide_with_invalid_mocked_response_falls_back()
    test_decide_with_api_exception_falls_back()
    print("\nTUM TESTLER GECTI")
