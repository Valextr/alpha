"""Daily summary report generator.

Produces end-of-day reports in plain text and parquet formats.
Reports are saved to the configured output directory for archival.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl

from .config import MonitoringConfig
from .data_models import Alert, DailySummary
from .metrics import MetricsTracker


class ReportGenerator:
    """Generate daily summary reports.

    Usage:
        gen = ReportGenerator(config, tracker)
        summary = gen.generate_daily()
        gen.save(summary)
    """

    def __init__(
        self,
        config: MonitoringConfig,
        tracker: MetricsTracker,
    ):
        self.config = config
        self.tracker = tracker
        self._reports: list[DailySummary] = []

    def generate_daily(self, date_str: Optional[str] = None) -> DailySummary:
        """Generate a DailySummary from current metrics state."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        summary = self.tracker.daily_summary(date_str)
        self._reports.append(summary)
        return summary

    def save(self, summary: DailySummary, format: str = "text") -> Path:
        """Save a report to disk.

        Args:
            summary: The daily summary to save.
            format: Output format — "text" (default), "parquet".

        Returns:
            Path to the saved report file.
        """
        filename = f"daily_report_{summary.date}.{format}"
        filepath = self.config.report_output_dir / filename

        if format == "text":
            self._save_text(summary, filepath)
        elif format == "parquet":
            self._save_parquet(summary, filepath)
        else:
            raise ValueError(f"Unknown report format: {format}")

        return filepath

    def _save_text(self, summary: DailySummary, path: Path) -> None:
        """Save report as plain text."""
        lines = [
            f"=== Alpha Daily Report: {summary.date} ==",
            "",
            f"Starting Balance:  ${summary.starting_balance:>14,.2f}",
            f"Ending Balance:    ${summary.ending_balance:>14,.2f}",
            f"Daily P&L:         [{'' if summary.daily_pnl >= 0 else '-'}${abs(summary.daily_pnl):>12,.2f}]",
            f"Daily Return:      {summary.daily_return:>10.2%}",
            "",
            f"Trades:            {summary.num_trades}",
            f"Wins:              {summary.num_wins}",
            f"Losses:            {summary.num_losses}",
            f"Win Rate:          {summary.win_rate:>10.1%}",
            "",
            f"Current Drawdown:  {summary.current_drawdown:>10.2%}",
            f"Max Drawdown:      {summary.max_drawdown:>10.2%}",
            f"Sharpe Ratio:      {summary.sharpe_ratio:>12.3f}",
            "",
            f"Open Positions:    {summary.positions_count}",
            f"Alerts Today:      {summary.alerts_count}",
        ]
        path.write_text("\n".join(lines) + "\n")

    def _save_parquet(self, summary: DailySummary, path: Path) -> None:
        """Save report as parquet for programmatic access."""
        pl.DataFrame([summary.model_dump()]).write_parquet(path)

    def save_all_reports(self) -> Path:
        """Save all accumulated reports as a single parquet file."""
        if not self._reports:
            return self.config.report_output_dir / "no_reports.parquet"

        path = self.config.report_output_dir / "all_reports.parquet"
        pl.DataFrame([r.model_dump() for r in self._reports]).write_parquet(path)
        return path

    def get_reports(self) -> list[DailySummary]:
        """Return all generated reports."""
        return self._reports.copy()