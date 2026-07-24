"""Shared helpers for building monthly forecast horizons."""

from __future__ import annotations

import pandas as pd


def build_future_dates(
    last_real_date: pd.Timestamp,
    minimum_horizon_months: int = 12,
    forecast_end_date: str | None = None,
) -> pd.DatetimeIndex:
    """Return future month starts with a minimum horizon and optional end month.

    The configured end date is a floor rather than a fixed horizon. When the
    latest real month advances, the function still guarantees at least
    ``minimum_horizon_months`` predictions.
    """
    if minimum_horizon_months < 1:
        raise ValueError("minimum_horizon_months must be at least 1.")

    last_month = pd.Timestamp(last_real_date).to_period("M").to_timestamp()
    minimum_end = (
        last_month + pd.DateOffset(months=minimum_horizon_months)
    ).to_period("M").to_timestamp()
    end_date = minimum_end

    if forecast_end_date:
        configured_end = pd.Timestamp(forecast_end_date).to_period("M").to_timestamp()
        end_date = max(end_date, configured_end)

    return pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        end=end_date,
        freq="MS",
    )
