"""Terminal-based monitoring dashboard using Rich.

Provides a live-updating terminal dashboard showing:
- Portfolio overview (balance, P&L, drawdown)
- Position table
- Recent fills
- Active alerts
- Daily metrics

Usage:
    dashboard = Dashboard(config)
    dashboard.update(metrics, positions, fills, alerts)
    dashboard.render()  # blocks with auto-refresh
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

from .config import MonitoringConfig
from .data_models import Fill, Position, Alert, AlertSeverity


class Dashboard:
    """Terminal-based monitoring dashboard.

    Renders a live-updating dashboard in the terminal using Rich.
    The dashboard is updated by calling update() with current metrics,
    positions, fills, and alerts.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.console = Console()
        self._metrics: dict = {}
        self._positions: list[Position] = []
        self._fills: list[Fill] = []
        self._alerts: list[Alert] = []

    def update(
        self,
        metrics: dict,
        positions: list[Position] | None = None,
        fills: list[Fill] | None = None,
        alerts: list[Alert] | None = None,
    ) -> None:
        """Update dashboard with current state."""
        self._metrics = metrics
        if positions is not None:
            self._positions = positions
        if fills is not None:
            self._fills = fills
        if alerts is not None:
            self._alerts = alerts

    def render(self) -> str:
        """Render the dashboard and return as a string."""

        with self.console.capture() as capture:
            self.console.print(self._overview_panel())
            self.console.print()

            # Positions and alerts rendered side-by-side via text
            self.console.print(
                Columns(
                    [self._positions_panel(), self._alerts_panel()],
                    equal=True,
                )
            )
            self.console.print()
            self.console.print(self._fills_panel())

        return capture.get()

    def _color_for_value(self, value: float, positive_is_good: bool = True) -> str:
        """Return green/red color based on value sign."""
        if positive_is_good:
            return "green" if value >= 0 else "red"
        return "red" if value > 0.1 else ("yellow" if value > 0.05 else "green")

    def _overview_panel(self) -> Panel:
        """Portfolio overview panel."""
        m = self._metrics

        total_pnl = m.get("total_pnl", 0.0)
        pnl_color = "green" if total_pnl >= 0 else "red"

        drawdown = m.get("drawdown", 0.0)
        dd_color = self._color_for_value(drawdown, positive_is_good=False)

        lines = [
            f"[bold cyan]Portfolio Overview[/bold cyan]  "
            f"[dim]{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/dim]",
            f"Balance:        [bold]{m.get('balance', 0):>12,.2f}[/bold]",
            f"Total P&L:      [{pnl_color}]{total_pnl:>12,.2f}[/]",
            f"Total Return:   [{pnl_color}]{m.get('total_return', 0):>11.2%}[/]",
            f"Realized P&L:   [{'green' if m.get('realized_pnl', 0) >= 0 else 'red'}]{m.get('realized_pnl', 0):>12,.2f}[/]",
            f"Unrealized P&L: [{'green' if m.get('unrealized_pnl', 0) >= 0 else 'red'}]{m.get('unrealized_pnl', 0):>12,.2f}[/]",
            "",
            f"Drawdown:       [{dd_color}]{drawdown:>11.2%}[/]",
            f"Max Drawdown:   [{dd_color}]{m.get('max_drawdown', 0):>11.2%}[/]",
            f"Sharpe Ratio:   [bold]{m.get('sharpe_ratio', 0):>12.3f}[/bold]",
            f"Win Rate:       [bold]{m.get('win_rate', 0):>11.1%}[/bold]",
            "",
            f"Trades:         {m.get('num_trades', 0):>12}",
            f"Open Positions: {m.get('num_positions', 0):>12}",
        ]

        return Panel(
            "\n".join(lines),
            title="[bold]ALPHA MONITORING[/bold]",
            border_style="cyan",
            box=box.SIMPLE,
        )

    def _positions_panel(self) -> Panel:
        """Positions table panel."""
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Qty", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Unrealized", justify="right")

        positions = sorted(
            self._positions,
            key=lambda p: abs(p.market_value),
            reverse=True,
        )

        for pos in positions[:20]:  # Max 20 positions shown
            if pos.quantity == 0:
                continue
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                pos.ticker,
                str(pos.quantity),
                f"{pos.avg_cost:,.2f}",
                f"{pos.current_price:,.2f}",
                f"[{pnl_color}]{pos.unrealized_pnl:>10,.2f}[/]",
            )

        if not positions:
            table.add_row("[dim]No open positions[/dim]", "", "", "", "")

        return Panel(table, title="[bold]Positions[/bold]", border_style="blue")

    def _alerts_panel(self) -> Panel:
        """Alerts panel."""
        if not self._alerts:
            content = Text("No active alerts", style="green italic")
            return Panel(content, title="[bold]Alerts[/bold]", border_style="green")

        # Sort by severity (CRITICAL first)
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.INFO: 2,
        }
        sorted_alerts = sorted(
            self._alerts,
            key=lambda a: severity_order.get(a.severity, 99),
        )

        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Severity", style="bold")
        table.add_column("Type", no_wrap=True)
        table.add_column("Message", overflow="fold")

        severity_styles = {
            AlertSeverity.CRITICAL: "bold red",
            AlertSeverity.WARNING: "yellow",
            AlertSeverity.INFO: "blue",
        }

        for alert in sorted_alerts[-20:]:  # Max 20 alerts shown
            style = severity_styles.get(alert.severity, "white")
            ack = " ✓" if alert.acknowledged else ""
            table.add_row(
                f"[{style}]{alert.severity.value}[/]",
                f"[{style}]{alert.alert_type.value}[/]",
                f"{alert.message}{ack}",
            )

        return Panel(table, title="[bold]Alerts[/bold]", border_style="red")

    def _fills_panel(self) -> Panel:
        """Recent fills table."""
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Time", style="dim", no_wrap=True)
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Side", justify="center")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Commission", justify="right")

        fills = sorted(self._fills, key=lambda f: f.timestamp, reverse=True)

        for fill in fills[:15]:  # Max 15 recent fills
            side_color = "green" if fill.side.value == "BUY" else "red"
            time_str = fill.timestamp.strftime("%H:%M:%S")
            table.add_row(
                time_str,
                fill.ticker,
                f"[{side_color}]{fill.side.value}[/]",
                str(fill.quantity),
                f"{fill.price:,.2f}",
                f"{fill.commission:,.2f}" if fill.commission else "0.00",
            )

        if not fills:
            table.add_row("[dim]No fills recorded[/dim]", "", "", "", "", "")

        return Panel(table, title="[bold]Recent Fills[/bold]", border_style="dim blue")