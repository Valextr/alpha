"""Parameter perturbation tests.

Purpose:
    Verify the strategy is robust — not overfitted to a narrow set of
    parameter values. If a ±20% change in a lookback window destroys the
    IC, the signal is probably fitting noise.

Approach:
    1. Scan all registered signals for integer-tunable parameters
       (lookback, formation_days, etc.).
    2. For each parameter, generate perturbed values at -20%, -10%,
       baseline, +10%, +20%.
    3. Re-run the signal with each perturbed value.
    4. Compare IC and win rate against baseline.
    5. Flag parameters where a small change causes large metric drops.

Design decisions:
    - Perturbations are computed by direct function re-invocation with
      modified parameters — no registry mutation.
    - Metrics are computed on the SAME data split for every variant,
      so the comparison is fair.
    - A signal is "fragile" if the IC drops by >50% from baseline at
      any perturbation level.
    - A signal is "robust" if IC stays within 80% of baseline across
      all perturbation levels.
    - Everything else is "sensitive" (between fragile and robust).

See: López de Prado, "Advances in Financial Machine Learning,"
     Ch. 3: Cross-Sectional Mean and Variance Deflated Performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from src.signals.base import rank_ic, win_rate
from src.signals.registry import registry


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class PerturbationResult:
    """Result of a single perturbation test for one parameter value."""

    signal_name: str
    parameter: str
    baseline_value: int
    perturbed_value: int
    pct_change: float  # e.g. -20.0, -10.0, 0.0, 10.0, 20.0

    # Metrics at this perturbation level
    ic: float
    win_rate: float
    signal_std: float
    signal_mean: float

    # Relative to baseline
    ic_change_pct: float | None = None  # None for the baseline row itself
    win_rate_change_pct: float | None = None

    def __post_init__(self):
        if self.ic_change_pct is not None:
            # Negative value means IC dropped vs baseline
            pass


@dataclass
class SignalPerturbationReport:
    """Full perturbation report for one signal + one parameter."""

    signal_name: str
    parameter: str
    baseline_value: int
    baseline_ic: float
    baseline_win_rate: float

    # All perturbation results (includes baseline row)
    results: list[PerturbationResult] = field(default_factory=list)

    # Robustness classification
    status: str = ""  # "robust", "sensitive", "fragile"
    worst_ic_drop_pct: float = 0.0  # how much IC dropped at worst

    def classify(self) -> str:
        """Classify robustness based on worst-case IC drop.

        - Robust:  IC stays within 80% of baseline at all levels
        - Fragile: IC drops by >50% from baseline at any level
        - Sensitive: everything in between
        """
        if not self.results:
            return "no_data"

        drops = []
        for r in self.results:
            if r.ic_change_pct is not None:
                drops.append(r.ic_change_pct)

        if not drops:
            self.worst_ic_drop_pct = 0.0
            self.status = "robust"
            return self.status

        self.worst_ic_drop_pct = min(drops)

        if self.worst_ic_drop_pct >= -20.0:
            self.status = "robust"
        elif self.worst_ic_drop_pct <= -50.0:
            self.status = "fragile"
        else:
            self.status = "sensitive"

        return self.status


@dataclass
class PerturbationSummary:
    """Top-level summary across all signals and parameters."""

    total_signals_tested: int = 0
    total_parameters_tested: int = 0
    robust_count: int = 0
    sensitive_count: int = 0
    fragile_count: int = 0

    reports: list[SignalPerturbationReport] = field(default_factory=list)

    def add_report(self, report: SignalPerturbationReport) -> None:
        self.reports.append(report)
        self.total_parameters_tested += 1
        if report.classify() == "robust":
            self.robust_count += 1
        elif report.classify() == "sensitive":
            self.sensitive_count += 1
        elif report.classify() == "fragile":
            self.fragile_count += 1

    def flag_fragile(self) -> list[str]:
        """Return descriptions of all fragile parameters."""
        return [
            f"{r.signal_name}.{r.parameter} (baseline={r.baseline_value}, "
            f"worst IC drop: {r.worst_ic_drop_pct:+.1f}%)"
            for r in self.reports if r.status == "fragile"
        ]

    def flag_sensitive(self) -> list[str]:
        """Return descriptions of all sensitive parameters."""
        return [
            f"{r.signal_name}.{r.parameter} (baseline={r.baseline_value}, "
            f"worst IC drop: {r.worst_ic_drop_pct:+.1f}%)"
            for r in self.reports if r.status == "sensitive"
        ]


# ── Parameter discovery ──────────────────────────────────────────────


TUNABLE_PARAM_NAMES = {
    "lookback",
    "formation_days",
    "skip_days",
    "window",
    "horizon",
    "fast_period",
    "slow_period",
}


def _get_tunable_params(signal_name: str) -> dict[str, int]:
    """Extract tunable integer parameters from a signal's metadata.

    Only parameters whose names match known tunable patterns are returned.
    Boolean flags and lists are excluded — we perturb numeric windows,
    not on/off toggles.
    """
    meta = registry.get_signal(signal_name)
    if meta is None:
        return {}

    tunables: dict[str, int] = {}
    for key, value in meta.parameters.items():
        if key in TUNABLE_PARAM_NAMES and isinstance(value, int) and value > 1:
            tunables[key] = value
    return tunables


def _perturbed_values(baseline: int, pcts: list[float] | None = None) -> list[int]:
    """Generate perturbed values around a baseline.

    Default pcts: [-20, -10, 0, +10, +20].
    Values are clamped to >= 1.
    """
    if pcts is None:
        pcts = [-20.0, -10.0, 0.0, 10.0, 20.0]

    values = []
    for pct in pcts:
        val = max(1, round(baseline * (1.0 + pct / 100.0)))
        values.append(val)
    return values


# ── Signal re-generation with perturbed parameters ───────────────────


def _recreate_signal(
    df: pl.DataFrame,
    signal_name: str,
    param_name: str,
    param_value: int,
) -> pl.DataFrame:
    """Run a signal generator with a modified parameter value.

    The standard registry functions use hardcoded values (e.g., lookback=21).
    We extract the internal _generate_* function and call it directly with
    the perturbed parameter.

    This approach avoids mutating the global registry and is fully
    reproducible.
    """
    # Import the module dynamically
    if signal_name.startswith("mean_reversion"):
        from src.signals import mean_reversion  # noqa: F811

        return mean_reversion._generate_mean_reversion(
            df,
            lookback=param_value,
            out_col=f"signal_{signal_name}_perturbed",
        )
    elif signal_name.startswith("momentum"):
        from src.signals import momentum  # noqa: F811

        # Momentum signals use pre-computed return columns (return_21d, return_63d).
        # When perturbing formation_days, we need to compute the return for the
        # perturbed window dynamically, since the pre-computed feature may not match.
        return_col = f"return_{param_value}d"
        if return_col not in df.columns:
            # Compute the return column on-the-fly:
            # return_Nd = (close / close.shift(N)) - 1, per ticker
            close = pl.col("close")
            shifted = close.shift(param_value).over("ticker")
            df = df.with_columns(
                ((close / shifted) - 1).alias(return_col)
            )

        return momentum._generate_momentum(
            df,
            return_col=return_col,
            out_col=f"signal_{signal_name}_perturbed",
        )
    else:
        # Fallback: try the internal function with the param override
        meta = registry.get_signal(signal_name)
        if meta is None:
            raise ValueError(f"Unknown signal: {signal_name}")

        gen_fn = registry.get_generate_fn(signal_name)
        if gen_fn is None:
            raise ValueError(f"No generator for signal: {signal_name}")

        # For signals we don't have special handling for, try calling
        # the standard generator. If the param isn't directly tunable
        # via the function signature, the result may not actually vary.
        # This is a best-effort approach for future signals.
        return gen_fn(df)


# ── Core perturbation test ──────────────────────────────────────────


def run_perturbation_test(
    df: pl.DataFrame,
    signal_name: str,
    param_name: str,
    baseline_value: int,
    target_col: str = "forward_return_1",
    pcts: list[float] | None = None,
) -> SignalPerturbationReport:
    """Run a perturbation test for one signal and one parameter.

    Args:
        df: Feature-enriched DataFrame with forward return columns.
            Must be sorted by (ticker, date).
        signal_name: Registered signal name (e.g., "mean_reversion_21d").
        param_name: Parameter to perturb (e.g., "lookback").
        baseline_value: Current parameter value.
        target_col: Forward return column to compute IC against.
        pcts: List of percentage changes to test (default: ±20%).

    Returns:
        SignalPerturbationReport with all perturbation results.
    """
    perturb_vals = _perturbed_values(baseline_value, pcts)

    # Compute baseline metrics first
    baseline_df = _recreate_signal(df, signal_name, param_name, baseline_value)
    baseline_col = f"signal_{signal_name}_perturbed"

    baseline_ic = rank_ic(baseline_col, target_col, baseline_df)
    baseline_wr = win_rate(baseline_col, target_col, baseline_df)

    report = SignalPerturbationReport(
        signal_name=signal_name,
        parameter=param_name,
        baseline_value=baseline_value,
        baseline_ic=baseline_ic,
        baseline_win_rate=baseline_wr,
    )

    for i, pv in enumerate(perturb_vals):
        pct = pcts[i] if pcts else (-20.0 + i * 10.0)

        if pv == baseline_value:
            # Baseline row — no change metrics
            col_vals = baseline_df[baseline_col].drop_nulls()
            result = PerturbationResult(
                signal_name=signal_name,
                parameter=param_name,
                baseline_value=baseline_value,
                perturbed_value=pv,
                pct_change=pct,
                ic=baseline_ic,
                win_rate=baseline_wr,
                signal_std=float(col_vals.std()) if len(col_vals) > 0 else 0.0,
                signal_mean=float(col_vals.mean()) if len(col_vals) > 0 else 0.0,
            )
        else:
            # Perturbed run
            pert_df = _recreate_signal(df, signal_name, param_name, pv)
            pert_col = f"signal_{signal_name}_perturbed"

            ic = rank_ic(pert_col, target_col, pert_df)
            wr = win_rate(pert_col, target_col, pert_df)
            col_vals = pert_df[pert_col].drop_nulls()

            ic_change = ((ic - baseline_ic) / abs(baseline_ic) * 100.0) if baseline_ic != 0 else 0.0
            wr_change = ((wr - baseline_wr) / abs(baseline_wr) * 100.0) if baseline_wr != 0 else 0.0

            result = PerturbationResult(
                signal_name=signal_name,
                parameter=param_name,
                baseline_value=baseline_value,
                perturbed_value=pv,
                pct_change=pct,
                ic=ic,
                win_rate=wr,
                signal_std=float(col_vals.std()) if len(col_vals) > 0 else 0.0,
                signal_mean=float(col_vals.mean()) if len(col_vals) > 0 else 0.0,
                ic_change_pct=ic_change,
                win_rate_change_pct=wr_change,
            )

        report.results.append(result)

    return report


# ── Full sweep across all signals ────────────────────────────────────


def run_full_perturbation_sweep(
    df: pl.DataFrame,
    target_col: str = "forward_return_1",
    pcts: list[float] | None = None,
) -> PerturbationSummary:
    """Run perturbation tests across all registered signals.

    For each signal, discovers tunable integer parameters and runs
    a ±20% perturbation sweep.

    Args:
        df: Feature-enriched DataFrame with forward returns.
        target_col: Forward return column for IC computation.
        pcts: Percentage changes to test (default: ±20%).

    Returns:
        PerturbationSummary with all reports and classification.
    """
    summary = PerturbationSummary()

    all_signals = registry.list_signals()
    summary.total_signals_tested = len(all_signals)

    for meta in all_signals:
        tunables = _get_tunable_params(meta.name)
        if not tunables:
            continue

        for param_name, baseline_value in tunables.items():
            report = run_perturbation_test(
                df=df,
                signal_name=meta.name,
                param_name=param_name,
                baseline_value=baseline_value,
                target_col=target_col,
                pcts=pcts,
            )
            summary.add_report(report)

    return summary


# ── Reporting ────────────────────────────────────────────────────────


def format_perturbation_report(summary: PerturbationSummary) -> str:
    """Format the perturbation summary as a readable text report.

    Args:
        summary: Output of run_full_perturbation_sweep().

    Returns:
        Human-readable report string.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PERTURBATION TEST REPORT")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"Signals tested : {summary.total_signals_tested}")
    lines.append(f"Parameters tested: {summary.total_parameters_tested}")
    lines.append(f"Robust         : {summary.robust_count}")
    lines.append(f"Sensitive      : {summary.sensitive_count}")
    lines.append(f"Fragile        : {summary.fragile_count}")
    lines.append("")

    for report in summary.reports:
        status_icon = {"robust": "[OK]", "sensitive": "[WARN]", "fragile": "[FAIL]"}.get(
            report.status, "[?]"
        )
        lines.append(f"{status_icon} {report.signal_name}.{report.parameter}")
        lines.append(f"      Baseline: {report.baseline_value} (IC={report.baseline_ic:.4f}, "
                     f"WR={report.baseline_win_rate:.2%})")

        for r in report.results:
            tag = ""
            if r.ic_change_pct is not None:
                tag = f"  IC change: {r.ic_change_pct:+.1f}%"
            lines.append(
                f"        {r.perturbed_value:4d} ({r.pct_change:+.0f}%) "
                f"  IC={r.ic:.4f}  WR={r.win_rate:.2%}  std={r.signal_std:.4f}{tag}"
            )

        lines.append(
            f"      Status: {report.status.upper()} "
            f"(worst IC drop: {report.worst_ic_drop_pct:+.1f}%)"
        )
        lines.append("")

    if summary.flag_fragile():
        lines.append("-" * 72)
        lines.append("FRAGILE PARAMETERS (IC drops >50%):")
        for desc in summary.flag_fragile():
            lines.append(f"  ! {desc}")
        lines.append("")

    if summary.flag_sensitive():
        lines.append("-" * 72)
        lines.append("SENSITIVE PARAMETERS (IC drops 20-50%):")
        for desc in summary.flag_sensitive():
            lines.append(f"  ~ {desc}")
        lines.append("")

    if summary.robust_count == summary.total_parameters_tested and summary.total_parameters_tested > 0:
        lines.append("All parameters are robust. Strategy appears stable.")

    lines.append("=" * 72)
    return "\n".join(lines)