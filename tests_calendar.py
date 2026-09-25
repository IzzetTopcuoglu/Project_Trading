"""
economic_calendar.py ve dataset.py entegrasyonu icin test dosyasi
(sentetik veriyle, agdan/DuckDB'den bagimsiz calisir bolumler ayri).
Proje kokunde calistir:
    python tests_calendar.py
"""
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.economic_calendar import add_calendar_features, get_calendar_context


def test_add_calendar_features_known_dates():
    # 2026-04-10 bilinen bir CPI gunu (bkz. economic_calendar.py._CPI_RELEASE_DAYS)
    timestamps = pd.to_datetime(
        ["2026-04-08 10:00", "2026-04-09 10:00", "2026-04-10 03:00", "2026-04-11 10:00"], utc=True
    )
    df = pd.DataFrame({"datetime": timestamps, "close": [1.0, 2.0, 3.0, 4.0]})
    out = add_calendar_features(df, window_days=1)

    # 08 Nisan: bir sonraki olaya (10 Nisan) 2 gun var -> window=1 icinde degil
    assert out.loc[0, "calendar_days_to_next_event"] == 2
    assert out.loc[0, "calendar_high_impact_soon"] == 0

    # 09 Nisan: 10 Nisan'a 1 gun var -> window=1 icinde
    assert out.loc[1, "calendar_days_to_next_event"] == 1
    assert out.loc[1, "calendar_high_impact_soon"] == 1

    # 10 Nisan: olayin kendi gunu -> 0 gun
    assert out.loc[2, "calendar_days_to_next_event"] == 0
    assert out.loc[2, "calendar_high_impact_soon"] == 1

    print("test_add_calendar_features_known_dates OK ->\n", out[["datetime", "calendar_days_to_next_event", "calendar_high_impact_soon"]])


def test_add_calendar_features_no_nan_no_crash():
    # genis bir tarih araligi (bazi tarihler listenin disina da tasabilir)
    timestamps = pd.date_range("2025-01-01", "2026-12-31", freq="17D", tz="UTC")
    df = pd.DataFrame({"datetime": timestamps, "close": np.random.rand(len(timestamps))})
    out = add_calendar_features(df)
    assert out["calendar_days_to_next_event"].isna().sum() == 0
    assert out["calendar_high_impact_soon"].isin([0, 1]).all()
    print(f"test_add_calendar_features_no_nan_no_crash OK -> {len(out)} satir, NaN yok")


def test_get_calendar_context_known_date():
    # 2026-04-09 -> bir sonraki bilinen olay 2026-04-10 (CPI), 1 gun kaldi
    now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    ctx = get_calendar_context(now=now, window_days=1)
    assert ctx["next_event_type"] == "CPI", ctx
    assert ctx["next_event_date"] == "2026-04-10", ctx
    assert ctx["days_until_next_event"] == 1, ctx
    assert ctx["high_impact_soon"] is True, ctx
    print("test_get_calendar_context_known_date OK ->", ctx)


def test_get_calendar_context_far_from_event():
    now = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    ctx = get_calendar_context(now=now, window_days=1)
    assert ctx["high_impact_soon"] is False, ctx
    assert ctx["days_until_next_event"] > 1, ctx
    print("test_get_calendar_context_far_from_event OK ->", ctx)


def _make_synthetic_ohlcv(n=3000, start="2026-03-01", freq="5min"):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    price = 60000 + np.cumsum(rng.normal(0, 30, size=n))
    price = np.maximum(price, 1000)
    df = pd.DataFrame({
        "timestamp": (ts.view("int64") // 1_000_000),
        "open": price,
        "high": price * (1 + rng.uniform(0, 0.002, size=n)),
        "low": price * (1 - rng.uniform(0, 0.002, size=n)),
        "close": price + rng.normal(0, 10, size=n),
        "volume": rng.uniform(1, 100, size=n),
        "datetime": ts,
    })
    return df


def test_prepare_dataset_with_calendar():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        import duckdb

        db_path = tmp_dir / "test.duckdb"
        df = _make_synthetic_ohlcv()
        con = duckdb.connect(str(db_path))
        con.register("df_view", df)
        con.execute("CREATE TABLE ohlcv_5m AS SELECT * FROM df_view")
        con.close()

        with patch("src.config.DUCKDB_PATH", db_path), patch("src.config.OHLCV_TABLE", "ohlcv_5m"):
            from src.models.lstm_price.dataset import prepare_dataset, CALENDAR_FEATURE_COLUMNS as PRICE_CAL_COLS
            from src.models.lstm_direction.dataset import prepare_direction_dataset

            split = prepare_dataset(include_calendar=True)
            assert split.X_train.shape[2] == 7 + len(PRICE_CAL_COLS), split.X_train.shape
            assert not np.isnan(split.X_train).any(), "X_train icinde NaN olmamali"

            dsplit = prepare_direction_dataset(include_calendar=True)
            assert dsplit.X_train.shape[2] == 7 + len(PRICE_CAL_COLS), dsplit.X_train.shape
            assert not np.isnan(dsplit.X_train).any()

        print(
            "test_prepare_dataset_with_calendar OK -> "
            f"lstm_price X_train {split.X_train.shape}, lstm_direction X_train {dsplit.X_train.shape}"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_add_calendar_features_known_dates()
    test_add_calendar_features_no_nan_no_crash()
    test_get_calendar_context_known_date()
    test_get_calendar_context_far_from_event()
    test_prepare_dataset_with_calendar()
    print("\nTUM TESTLER GECTI")
