"""Tests for the IBKR execution client.

These tests verify the client module loads correctly, configuration
is wired properly, and contract/order helpers produce valid objects.

Note: Connection tests (actual IB Gateway handshake) are skipped
unless IBKR_TEST_CONNECT=true is set in the environment, since they
require a running IB Gateway instance and valid credentials.
"""

import os
from unittest.mock import MagicMock, patch

from src.execution.ibkr_client import (
    IBKRClient,
    IBKRConfig,
    FillRecord,
    PositionRecord,
    HAS_IB,
)


class TestIBKRConfig:
    """Test configuration loading."""

    def test_default_config(self):
        """Default config uses sensible defaults."""
        cfg = IBKRConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 4001
        assert cfg.client_id == 1
        assert cfg.trade_mode == "PAPER"
        assert cfg.max_reconnect_attempts == 3

    def test_from_env(self):
        """Environment variables override defaults."""
        with patch.dict(os.environ, {
            "IBKR_HOST": "192.168.1.100",
            "IBKR_PORT": "7497",
            "IBKR_CLIENT_ID": "5",
            "IBKR_TRADE_MODE": "LIVE",
        }):
            cfg = IBKRConfig.from_env()
            assert cfg.host == "192.168.1.100"
            assert cfg.port == 7497
            assert cfg.client_id == 5
            assert cfg.trade_mode == "LIVE"

    def test_trade_mode_uppercase(self):
        """Trade mode is normalized to uppercase by the client."""
        cfg = IBKRConfig(trade_mode="paper")
        client = IBKRClient(config=cfg)
        assert client.config.trade_mode == "PAPER"


class TestIBKRClient:
    """Test client initialization and helpers (no connection)."""

    def test_client_creation(self):
        """Client initializes without credentials."""
        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        assert not client.connected
        assert client.account == ""

    def test_client_repr(self):
        """String representation includes connection state."""
        cfg = IBKRConfig(trade_mode="PAPER")
        client = IBKRClient(config=cfg)
        repr_str = repr(client)
        assert "disconnected" in repr_str
        assert "PAPER" in repr_str

    def test_stock_contract(self):
        """Stock helper creates a valid contract."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        contract = client.stock("AAPL")
        assert contract.symbol == "AAPL"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"

    def test_forex_contract(self):
        """Forex helper creates a valid contract."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        contract = client.forex("EURUSD")
        assert contract.symbol == "EUR"

    def test_market_order(self):
        """Market order has correct fields."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        order = client.market_order("BUY", 100)
        assert order.action == "BUY"
        assert order.orderType == "MKT"
        assert order.totalQuantity == 100

    def test_limit_order(self):
        """Limit order has correct fields."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        order = client.limit_order("SELL", 50, limit_price=150.00)
        assert order.action == "SELL"
        assert order.orderType == "LMT"
        assert order.totalQuantity == 50
        assert order.lmtPrice == 150.00
        assert order.tif == "GTC"

    def test_stop_order(self):
        """Stop order has correct fields."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        order = client.stop_order("SELL", 25, stop_price=140.00)
        assert order.action == "SELL"
        assert order.orderType == "STP"
        assert order.totalQuantity == 25
        assert order.auxPrice == 140.00

    def test_bracket_order(self):
        """Bracket order returns three orders."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        orders = client.bracket_order(
            quantity=100,
            limit_price=150.00,
            take_profit=160.00,
            stop_loss=145.00,
        )
        assert len(orders) == 3
        parent, tp, sl = orders
        assert parent.action == "BUY"
        assert tp.action == "SELL"
        assert sl.action == "SELL"
        assert tp.ocaName == sl.ocaName

    def test_pre_order_check_invalid_quantity(self):
        """Zero/negative quantity is rejected."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        contract = client.stock("AAPL")
        order = client.market_order("BUY", 0)

        import pytest
        with pytest.raises(ValueError, match="Invalid order quantity"):
            client._pre_order_check(contract, order)

    def test_pre_order_check_fractional_shares(self):
        """Fractional shares are rejected for stock orders."""
        if not HAS_IB:
            import pytest
            pytest.skip("ib_async not installed")

        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        contract = client.stock("AAPL")
        order = client.market_order("BUY", 10.5)

        import pytest
        with pytest.raises(ValueError, match="whole shares"):
            client._pre_order_check(contract, order)

    def test_place_order_requires_connection(self):
        """Placing an order without connection raises ConnectionError."""
        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)
        contract = client.stock("AAPL")
        order = client.market_order("BUY", 100)

        import pytest
        with pytest.raises(ConnectionError, match="Not connected"):
            client.place_order(contract, order)

    def test_get_positions_requires_connection(self):
        """Getting positions without connection raises ConnectionError."""
        cfg = IBKRConfig()
        client = IBKRClient(config=cfg)

        import pytest
        with pytest.raises(ConnectionError, match="Not connected"):
            client.get_positions()


class TestFillRecord:
    """Test fill record dataclass."""

    def test_fill_record_creation(self):
        """FillRecord captures all fields."""
        fill = FillRecord(
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.00,
            date="2026-07-19",
            commission=0.65,
            currency="USD",
        )
        assert fill.symbol == "AAPL"
        assert fill.quantity == 100
        assert fill.commission == 0.65


class TestPositionRecord:
    """Test position record dataclass."""

    def test_position_record_creation(self):
        """PositionRecord captures all fields."""
        pos = PositionRecord(
            symbol="MSFT",
            quantity=50,
            avg_price=300.00,
            unrealized_pnl=250.00,
            realized_pnl=100.00,
        )
        assert pos.symbol == "MSFT"
        assert pos.quantity == 50
        assert pos.unrealized_pnl == 250.00