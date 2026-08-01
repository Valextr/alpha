"""Data quality validation checks."""

import logging
from datetime import date
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation: {len(self.errors)} errors, {len(self.warnings)} warnings",
        ]
        for err in self.errors:
            lines.append(f"  ERROR: {err}")
        for warn in self.warnings:
            lines.append(f"  WARN: {warn}")
        for info in self.info:
            lines.append(f"  INFO: {info}")
        return "\n".join(lines)


def check_negative_prices(df: pl.DataFrame, result: ValidationResult) -> ValidationResult:
    """Check for negative prices (should never happen)."""
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            neg_count = df.filter(pl.col(col) < 0).height
            if neg_count > 0:
                result.errors.append(
                    f"{neg_count} rows with negative {col} prices"
                )
    return result


def check_future_dates(df: pl.DataFrame, result: ValidationResult) -> ValidationResult:
    """Check for dates in the future."""
    today = date.today()
    future_count = df.filter(pl.col("date") > today).height
    if future_count > 0:
        result.errors.append(
            f"{future_count} rows with future dates (> {today})"
        )
    return result


def check_missing_dates(
    df: pl.DataFrame,
    result: Optional[ValidationResult] = None,
) -> ValidationResult:
    """Check for gaps in trading dates per ticker."""
    if result is None:
        result = ValidationResult()

    if "date" not in df.columns:
        result.warnings.append("No 'date' column for gap check")
        return result

    total_dates = df["date"].n_unique()
    result.info.append(f"Unique dates in dataset: {total_dates}")

    if "ticker" in df.columns:
        for ticker in df["ticker"].unique():
            ticker_dates = (
                df.filter(pl.col("ticker") == ticker)
                .select("date")
                .sort("date")
                .to_series()
                .to_list()
            )

            if len(ticker_dates) < 2:
                continue

            gaps = 0
            for i in range(1, len(ticker_dates)):
                delta = (ticker_dates[i] - ticker_dates[i - 1]).days
                if delta > 5:
                    gaps += 1

            if gaps > 0:
                result.warnings.append(
                    f"{ticker}: {gaps} large date gaps (>5 days)"
                )

    return result


def check_zero_volume(
    df: pl.DataFrame,
    result: Optional[ValidationResult] = None,
) -> ValidationResult:
    """Flag tickers with excessive zero-volume days."""
    if result is None:
        result = ValidationResult()

    if "volume" not in df.columns:
        result.warnings.append("No 'volume' column for zero-volume check")
        return result

    for ticker in df["ticker"].unique():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        zero_vol = ticker_df.filter(pl.col("volume") == 0).height
        total = ticker_df.height

        if total > 0 and (zero_vol / total) > 0.1:
            result.warnings.append(
                f"{ticker}: {zero_vol}/{total} ({zero_vol/total:.0%}) days with zero volume"
            )

    return result


def check_price_gaps(
    df: pl.DataFrame,
    threshold: float = 0.20,
    result: Optional[ValidationResult] = None,
) -> ValidationResult:
    """Check for suspicious price gaps (>20% day-over-day)."""
    if result is None:
        result = ValidationResult()

    if "adj_close" not in df.columns and "close" not in df.columns:
        result.warnings.append("No price column for gap check")
        return result

    price_col = "adj_close" if "adj_close" in df.columns else "close"
    suspicious = 0

    for ticker in df["ticker"].unique():
        prices = (
            df.filter(pl.col("ticker") == ticker)
            .sort("date")
            .select(price_col)
            .to_series()
            .to_list()
        )

        if len(prices) < 2:
            continue

        for i in range(1, len(prices)):
            if prices[i - 1] and prices[i]:
                gap = abs(prices[i] / prices[i - 1] - 1)
                if gap > threshold:
                    suspicious += 1

    if suspicious > 0:
        result.warnings.append(
            f"{suspicious} price gaps > {threshold:.0%} (may indicate splits/gaps)"
        )

    return result


def check_data_quality_flags(
    df: pl.DataFrame,
    result: Optional[ValidationResult] = None,
) -> ValidationResult:
    """Check data_quality column for issues."""
    if result is None:
        result = ValidationResult()

    if "data_quality" not in df.columns:
        result.info.append("No data_quality column (expected in silver/gold layers)")
        return result

    quality_counts = df["data_quality"].value_counts()
    for row in quality_counts.iter_rows(named=True):
        flag = row["data_quality"]
        count = row["count"]
        if flag == "good":
            continue
        result.warnings.append(f"data_quality='{flag}': {count} rows")

    return result


def run_all_checks(
    df: pl.DataFrame,
    layer: str = "silver",
) -> ValidationResult:
    """Run all validation checks on a DataFrame."""
    result = ValidationResult()

    result.info.append(f"Validating {layer} layer: {df.height} rows, {df.width} columns")

    # Always run
    result = check_negative_prices(df, result)
    result = check_future_dates(df, result)
    result = check_missing_dates(df, result=result)
    result = check_zero_volume(df, result=result)
    result = check_price_gaps(df, result=result)

    # Layer-specific
    if layer in ("silver", "gold"):
        result = check_data_quality_flags(df, result=result)

    return result


def main():
    """CLI entry point for validation."""
    import argparse
    import sys

    from .config import get_config

    parser = argparse.ArgumentParser(description="Validate data pipeline")
    parser.add_argument("layer", choices=["bronze", "silver", "gold"], nargs="?", default="silver")
    args = parser.parse_args()

    config = get_config()

    layer_dir = config.data_dir / args.layer / "daily"
    if not layer_dir.exists():
        print(f"Layer directory not found: {layer_dir}")
        sys.exit(1)

    try:
        df = pl.read_parquet(str(layer_dir / "**/*.parquet"))
    except Exception as e:
        print(f"Failed to read {args.layer} layer: {e}")
        sys.exit(1)

    result = run_all_checks(df, args.layer)
    print(result.summary())

    sys.exit(0 if result.is_valid else 1)
