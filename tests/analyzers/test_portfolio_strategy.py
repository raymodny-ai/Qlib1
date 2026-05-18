"""
组合优化与回测策略单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.analyzers.portfolio_strategy import (
    StrategyConfig,
    TopkDropoutStrategy,
    EqualWeightStrategy,
    ScoreWeightStrategy,
    PortfolioSimulator,
    RiskManager,
    WeightAllocator,
    WeightMethod,
    BacktestResult,
    TradeRecord,
    Position,
    BaseStrategy,
)


# ===== Fixtures =====

@pytest.fixture
def config():
    return StrategyConfig(
        top_k=10,
        min_k=5,
        dropout_threshold=0.2,
        rebalance_freq=5,
        weight_method=WeightMethod.EQUAL,
        commission_rate=0.001,
        slippage_bps=1.0,
        max_drawdown_limit=0.15,
        stop_loss=0.08,
    )


@pytest.fixture
def sample_scores():
    np.random.seed(42)
    instruments = [f"STOCK_{i:03d}" for i in range(50)]
    scores = pd.Series(np.random.randn(50), index=instruments)
    scores = scores.sort_values(ascending=False)
    return scores


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    instruments = [f"STOCK_{i:03d}" for i in range(50)]
    return pd.Series(np.random.uniform(10, 200, 50), index=instruments)


@pytest.fixture
def predictions_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    instruments = [f"STOCK_{i:03d}" for i in range(20)]
    data = np.random.randn(50, 20) * 0.02
    return pd.DataFrame(data, index=dates, columns=instruments)


@pytest.fixture
def prices_df():
    np.random.seed(99)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    instruments = [f"STOCK_{i:03d}" for i in range(20)]
    data = 50 + np.cumsum(np.random.randn(50, 20) * 0.5, axis=0)
    data = np.maximum(data, 1)
    return pd.DataFrame(data, index=dates, columns=instruments)


# ===== WeightAllocator 测试 =====

class TestWeightAllocator:

    def test_equal_weight(self):
        alloc = WeightAllocator(method=WeightMethod.EQUAL)
        scores = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        weights = alloc.allocate(scores)
        assert len(weights) == 3
        np.testing.assert_almost_equal(weights.sum(), 1.0)
        np.testing.assert_almost_equal(weights.iloc[0], 1/3)

    def test_score_weight(self):
        alloc = WeightAllocator(method=WeightMethod.SCORE)
        scores = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        weights = alloc.allocate(scores)
        np.testing.assert_almost_equal(weights.sum(), 1.0)
        assert weights["C"] >= weights["A"]  # 高分 = 高权重

    def test_score_weight_with_negatives(self):
        alloc = WeightAllocator(method=WeightMethod.SCORE)
        scores = pd.Series([-1.0, 0.0, 3.0], index=["A", "B", "C"])
        weights = alloc.allocate(scores)
        # 负分应归零
        assert weights["A"] == 0.0

    def test_score_sqrt_weight(self):
        alloc = WeightAllocator(method=WeightMethod.SCORE_SQRT)
        scores = pd.Series([1.0, 4.0, 9.0], index=["A", "B", "C"])
        weights = alloc.allocate(scores)
        np.testing.assert_almost_equal(weights.sum(), 1.0)
        # sqrt: C=3, B=2, A=1 → 权重比 3:2:1
        assert weights["C"] >= weights["B"] >= weights["A"]

    def test_rank_weight(self):
        alloc = WeightAllocator(method=WeightMethod.RANK)
        scores = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        weights = alloc.allocate(scores)
        np.testing.assert_almost_equal(weights.sum(), 1.0)

    def test_top_n_selection(self):
        alloc = WeightAllocator(method=WeightMethod.EQUAL)
        scores = pd.Series(np.arange(20), index=[f"S{i}" for i in range(20)])
        weights = alloc.allocate(scores, n_stocks=5)
        assert len(weights) == 5

    def test_max_weight_constraint(self):
        alloc = WeightAllocator(method=WeightMethod.EQUAL, max_weight=0.3)
        scores = pd.Series([1.0]*5, index=["A", "B", "C", "D", "E"])
        weights = alloc.allocate(scores)
        np.testing.assert_almost_equal(weights.sum(), 1.0)
        assert weights.max() <= 0.3 + 1e-10

    def test_empty_scores(self):
        alloc = WeightAllocator()
        weights = alloc.allocate(pd.Series(dtype=float))
        assert len(weights) == 0


# ===== RiskManager 测试 =====

class TestRiskManager:

    def test_init(self):
        rm = RiskManager(max_drawdown=0.15, stop_loss=0.08)
        assert rm.max_drawdown == 0.15
        assert rm.stop_loss == 0.08

    def test_drawdown_breach_no(self):
        rm = RiskManager(max_drawdown=0.15)
        assert not rm.check_drawdown_breach(1.0)
        assert not rm.check_drawdown_breach(0.9)
        assert abs(rm.current_drawdown - 0.1) < 1e-9

    def test_drawdown_breach_yes(self):
        rm = RiskManager(max_drawdown=0.15)
        rm.check_drawdown_breach(1.0)
        assert rm.check_drawdown_breach(0.8)

    def test_stop_loss_no_breach(self):
        rm = RiskManager(stop_loss=0.08)
        assert not rm.check_stop_loss("AAPL", 100.0, 95.0)

    def test_stop_loss_breach(self):
        rm = RiskManager(stop_loss=0.08)
        assert rm.check_stop_loss("AAPL", 100.0, 90.0)

    def test_stop_loss_already_stopped(self):
        rm = RiskManager(stop_loss=0.08)
        rm.check_stop_loss("AAPL", 100.0, 90.0)
        # 二次检查应仍为 True
        assert rm.check_stop_loss("AAPL", 100.0, 100.0)
        assert "AAPL" in rm.stopped_out_instruments

    def test_reset(self):
        rm = RiskManager(max_drawdown=0.15)
        rm.check_drawdown_breach(1.0)
        rm.check_stop_loss("AAPL", 100.0, 90.0)
        rm.reset()
        assert rm.current_drawdown == 0.0
        assert len(rm.stopped_out_instruments) == 0


# ===== TopkDropoutStrategy 测试 =====

class TestTopkDropoutStrategy:

    def test_init(self, config):
        strategy = TopkDropoutStrategy(config=config)
        assert strategy.config.top_k == 10

    def test_generate_weights(self, config, sample_scores, sample_prices):
        strategy = TopkDropoutStrategy(config=config)
        weights = strategy.generate_weights(
            sample_scores, {}, sample_prices, "2020-01-01"
        )
        assert isinstance(weights, pd.Series)
        assert len(weights) <= config.top_k
        np.testing.assert_almost_equal(weights.sum(), 1.0)

    def test_generate_weights_empty_scores(self, config, sample_prices):
        strategy = TopkDropoutStrategy(config=config)
        weights = strategy.generate_weights(
            pd.Series(dtype=float), {}, sample_prices, "2020-01-01"
        )
        assert len(weights) == 0

    def test_dropout_logic_sells_underperformers(self, config, sample_scores, sample_prices):
        """持有弱排名股票时应被淘汰"""
        strategy = TopkDropoutStrategy(config=config)

        # 创建持仓: 排名靠后的股票
        worst_instruments = sample_scores.nsmallest(5).index.tolist()
        positions = {}
        for inst in worst_instruments:
            positions[inst] = Position(
                instrument=inst,
                quantity=100,
                avg_cost=50.0,
                current_price=50.0,
                market_value=5000.0,
                weight=0.1,
                pnl=0.0,
            )

        weights = strategy.generate_weights(
            sample_scores, positions, sample_prices, "2020-01-01"
        )
        # 淘汰的股票不应在目标权重中
        for inst in worst_instruments:
            assert inst not in weights.index or weights.get(inst, 0) == 0

    def test_should_rebalance(self, config):
        strategy = TopkDropoutStrategy(config=config)
        assert strategy.should_rebalance("2020-01-01", 0)
        assert strategy.should_rebalance("2020-01-01", 5)
        assert not strategy.should_rebalance("2020-01-01", 1)

    def test_calculate_turnover(self, config):
        strategy = TopkDropoutStrategy(config=config)
        tw = pd.Series([0.5, 0.5], index=["A", "B"])
        cw = pd.Series([0.3, 0.7], index=["A", "B"])
        turnover = strategy.calculate_turnover(tw, cw)
        assert 0 <= turnover <= 1.0

    def test_calculate_turnover_identical(self, config):
        strategy = TopkDropoutStrategy(config=config)
        w = pd.Series([0.3, 0.3, 0.4], index=["A", "B", "C"])
        turnover = strategy.calculate_turnover(w, w)
        assert turnover == 0.0

    def test_apply_transaction_cost(self, config):
        strategy = TopkDropoutStrategy(config=config)
        comm, slip = strategy.apply_transaction_cost(10000, is_buy=True)
        assert comm == 10.0  # 0.001 * 10000
        assert slip == 1.0   # 1bp * 10000


# ===== EqualWeightStrategy 测试 =====

class TestEqualWeightStrategy:

    def test_equal_weights(self, config, sample_scores, sample_prices):
        strategy = EqualWeightStrategy(config=config)
        weights = strategy.generate_weights(
            sample_scores, {}, sample_prices, "2020-01-01"
        )
        assert len(weights) == config.top_k
        # 所有权重应相等
        assert len(set(weights.round(6))) == 1


# ===== ScoreWeightStrategy 测试 =====

class TestScoreWeightStrategy:

    def test_score_weighted(self, config, sample_scores, sample_prices):
        strategy = ScoreWeightStrategy(config=config)
        weights = strategy.generate_weights(
            sample_scores, {}, sample_prices, "2020-01-01"
        )
        assert len(weights) == config.top_k
        np.testing.assert_almost_equal(weights.sum(), 1.0)
        # 高分应获得高权重
        sorted_w = weights.sort_values(ascending=False)
        top_score_inst = sample_scores.index[0]
        assert top_score_inst in sorted_w.index[:3]


# ===== PortfolioSimulator 测试 =====

class TestPortfolioSimulator:

    def test_init(self, config):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=500000)
        assert sim.initial_capital == 500000
        assert sim.capital == 500000

    def test_run_basic(self, config, predictions_df, prices_df):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=1000000)
        result = sim.run(predictions_df, prices_df, verbose=False)

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 1000000
        assert result.total_trades >= 0
        assert len(result.daily_returns) > 0
        assert len(result.nav_curve) > 0

    def test_run_produces_nav_curve(self, config, predictions_df, prices_df):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=1000000)
        result = sim.run(predictions_df, prices_df, verbose=False)
        # 首日 NAV 接近初始资金 (扣除交易成本后略低)
        assert abs(result.nav_curve.iloc[0] - 1000000) < 10000

    def test_run_with_empty_data(self, config):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy)
        empty_pred = pd.DataFrame(index=pd.DatetimeIndex([]))
        empty_price = pd.DataFrame(index=pd.DatetimeIndex([]))
        result = sim.run(empty_pred, empty_price, verbose=False)
        assert result.total_return == 0

    def test_backtest_result_summary(self, config, predictions_df, prices_df):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=100000)
        result = sim.run(predictions_df, prices_df, verbose=False)
        summary = result.summary()
        assert "回测绩效报告" in summary
        assert "初始资金" in summary
        assert "夏普比率" in summary
        assert "最大回撤" in summary

    def test_transaction_costs_applied(self, config, predictions_df, prices_df):
        config.commission_rate = 0.01  # 高佣金便于观察
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=100000)
        result = sim.run(predictions_df, prices_df, verbose=False)
        # 有交易所以应有佣金
        if result.total_trades > 0:
            assert result.total_commission > 0

    def test_rebalance_frequency(self, predictions_df, prices_df):
        """低频调仓应减少交易次数"""
        config_freq = StrategyConfig(top_k=10, rebalance_freq=10)
        strategy_freq = TopkDropoutStrategy(config=config_freq)
        sim_freq = PortfolioSimulator(strategy_freq, initial_capital=100000)
        result_freq = sim_freq.run(predictions_df, prices_df, verbose=False)

        config_infreq = StrategyConfig(top_k=10, rebalance_freq=50)
        strategy_infreq = TopkDropoutStrategy(config=config_infreq)
        sim_infreq = PortfolioSimulator(strategy_infreq, initial_capital=100000)
        result_infreq = sim_infreq.run(predictions_df, prices_df, verbose=False)

        # 低频调仓不应产生更多交易
        assert result_freq.total_trades >= result_infreq.total_trades

    def test_nav_never_negative(self, config, predictions_df, prices_df):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=100000)
        result = sim.run(predictions_df, prices_df, verbose=False)
        assert (result.nav_curve > 0).all()

    def test_positions_history_recorded(self, config, predictions_df, prices_df):
        strategy = TopkDropoutStrategy(config=config)
        sim = PortfolioSimulator(strategy, initial_capital=100000)
        result = sim.run(predictions_df, prices_df, verbose=False)
        assert len(result.positions_history) > 0


# ===== StrategyConfig 测试 =====

class TestStrategyConfig:

    def test_default_config(self):
        config = StrategyConfig()
        assert config.top_k == 30
        assert config.commission_rate == 0.001
        assert config.slippage_bps == 1.0

    def test_custom_config(self):
        config = StrategyConfig(
            top_k=20,
            commission_rate=0.002,
        )
        assert config.top_k == 20
        assert config.commission_rate == 0.002
        assert config.slippage_bps == 1.0  # default


# ===== Position 数据类测试 =====

class TestPosition:

    def test_position_creation(self):
        pos = Position(
            instrument="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=155.0,
            market_value=15500.0,
            weight=0.1,
            pnl=500.0,
        )
        assert pos.instrument == "AAPL"
        assert pos.pnl == 500.0
        assert pos.holding_days == 0


# ===== TradeRecord 数据类测试 =====

class TestTradeRecord:

    def test_trade_record_creation(self):
        trade = TradeRecord(
            date="2020-01-15",
            instrument="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            commission=15.0,
            slippage_cost=1.5,
            notional=15000.0,
        )
        assert trade.action == "buy"
        assert trade.notional == 15000.0
