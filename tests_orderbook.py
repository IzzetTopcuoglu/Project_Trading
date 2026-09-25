"""
orderbook.py / log_orderbook.py icin gecici test dosyasi (agdan/Binance'ten
bagimsiz - sentetik veriyle). Proje kokunde calistir:
    python tests_orderbook.py
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.data.orderbook import compute_orderbook_imbalance, get_orderbook_context


def test_imbalance_balanced():
    ob = {"bids": [[100.0, 5.0], [99.9, 5.0]], "asks": [[100.1, 5.0], [100.2, 5.0]]}
    m = compute_orderbook_imbalance(ob, depth=2)
    assert abs(m["imbalance"]) < 1e-9, m
    assert m["best_bid"] == 100.0
    assert m["best_ask"] == 100.1
    assert abs(m["mid_price"] - 100.05) < 1e-9
    assert abs(m["spread"] - 0.1) < 1e-9
    print("test_imbalance_balanced OK ->", m)


def test_imbalance_buy_pressure():
    # alis hacmi satistan cok fazla -> imbalance pozitif olmali (yukselis sinyali)
    ob = {"bids": [[100.0, 30.0], [99.9, 20.0]], "asks": [[100.1, 5.0], [100.2, 5.0]]}
    m = compute_orderbook_imbalance(ob, depth=2)
    assert m["imbalance"] > 0.5, m
    print("test_imbalance_buy_pressure OK -> imbalance =", m["imbalance"])


def test_imbalance_sell_pressure():
    ob = {"bids": [[100.0, 2.0]], "asks": [[100.1, 40.0]]}
    m = compute_orderbook_imbalance(ob, depth=5)
    assert m["imbalance"] < -0.5, m
    print("test_imbalance_sell_pressure OK -> imbalance =", m["imbalance"])


def test_imbalance_empty_side_no_crash():
    # bir taraf tamamen bossa (ör. cok dusuk likidite) sifira bolme olmamali
    ob = {"bids": [], "asks": [[100.1, 5.0]]}
    m = compute_orderbook_imbalance(ob, depth=5)
    assert m["imbalance"] == -1.0, m
    assert m["best_bid"] is None
    assert m["mid_price"] is None
    print("test_imbalance_empty_side_no_crash OK ->", m)


def test_depth_is_respected():
    # depth=1 verilince sadece en iyi seviyeye bakmali, digerlerini yok saymali
    ob = {"bids": [[100.0, 10.0], [99.0, 1000.0]], "asks": [[100.1, 10.0], [101.0, 1000.0]]}
    m = compute_orderbook_imbalance(ob, depth=1)
    assert m["bid_volume"] == 10.0, m
    assert m["ask_volume"] == 10.0, m
    print("test_depth_is_respected OK ->", m)


def test_get_orderbook_context_shape():
    fake_ob = {"bids": [[100.0, 5.0]], "asks": [[100.1, 5.0]]}
    with patch("src.data.orderbook.fetch_orderbook_snapshot", return_value=fake_ob):
        ctx = get_orderbook_context(symbol="BTC/USDT", depth=5)
    for key in ["symbol", "timestamp", "imbalance", "best_bid", "best_ask", "mid_price", "spread", "spread_bps"]:
        assert key in ctx, f"eksik alan: {key}"
    assert ctx["symbol"] == "BTC/USDT"
    print("test_get_orderbook_context_shape OK ->", ctx)


def test_log_orderbook_appends_rows():
    # gercek .env/DUCKDB_PATH'i etkilememek icin gecici bir klasore yonlendiriyoruz
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        with patch("src.config.DUCKDB_PATH", tmp_dir / "test.duckdb"), \
             patch("src.config.ORDERBOOK_TABLE", "orderbook_snapshots"):
            from src.data.log_orderbook import append_snapshot
            import duckdb

            append_snapshot({"symbol": "BTC/USDT", "timestamp": "2026-01-01T00:00:00+00:00", "imbalance": 0.2})
            append_snapshot({"symbol": "BTC/USDT", "timestamp": "2026-01-01T00:05:00+00:00", "imbalance": -0.1})

            con = duckdb.connect(str(tmp_dir / "test.duckdb"), read_only=True)
            n = con.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0]
            con.close()
            assert n == 2, f"2 satir beklenirdi, {n} bulundu"
        print("test_log_orderbook_appends_rows OK -> 2 satir eklendi ve dogrulandi")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_imbalance_balanced()
    test_imbalance_buy_pressure()
    test_imbalance_sell_pressure()
    test_imbalance_empty_side_no_crash()
    test_depth_is_respected()
    test_get_orderbook_context_shape()
    test_log_orderbook_appends_rows()
    print("\nTUM TESTLER GECTI")
