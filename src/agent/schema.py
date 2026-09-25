"""
LLM Agent'in URETMESI GEREKEN karar JSON'unun semasi. Bu, Anthropic
API'sine "tool" (fonksiyon) olarak verilir - model bu fonksiyonu
cagirmaya ZORLANIR (tool_choice), yani serbest metin degil, bu semaya
UYAN bir JSON dondurmesi garanti altina alinir. Ayrica cagrildiktan
sonra jsonschema ile bir kez daha dogrulanir (savunma amacli - iki kat
kontrol).
"""
import jsonschema

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["long", "short", "no_trade"],
            "description": "long=yukselis beklentisiyle pozisyon ac, short=dusus beklentisiyle pozisyon ac, no_trade=bu turda islem acma",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Karar icin genel guven skoru (0-1)",
        },
        "leverage": {
            "type": "integer",
            "enum": [1, 5],
            "description": "Sinyaller (LSTM yonu, orderbook dengesizligi, takvim riski) UYUMLUYSA 5, KARISIKSA ya da no_trade ise 1",
        },
        "stop_limit_pct": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 5.0,
            "description": "Giris fiyatindan stop-limit'e kadar olan yuzde mesafe (ör. 1.0 = %1)",
        },
        "signals_aligned": {
            "type": "boolean",
            "description": "LSTM yonu, orderbook dengesizligi yonu ve takvim riski birbiriyle uyumlu mu",
        },
        "rationale": {
            "type": "string",
            "maxLength": 500,
            "description": "Kararin kisa gerekcesi (Turkce, 1-3 cumle)",
        },
    },
    "required": ["direction", "confidence", "leverage", "stop_limit_pct", "signals_aligned", "rationale"],
    "additionalProperties": False,
}

TOOL_NAME = "emit_trade_decision"

DECISION_TOOL = {
    "name": TOOL_NAME,
    "description": "BTC icin yapilandirilmis bir alim-satim karari bildir. Bu karar dogrudan otomatik olarak uygulanacak (insan onayi beklenmeyecek), o yuzden semaya tam uymali.",
    "input_schema": DECISION_SCHEMA,
}


def validate_decision(decision: dict) -> None:
    """Gecersizse jsonschema.exceptions.ValidationError firlatir."""
    jsonschema.validate(instance=decision, schema=DECISION_SCHEMA)
