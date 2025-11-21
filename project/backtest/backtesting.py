import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from itertools import product
import warnings

# 可选依赖：如果安装了 optuna 可以使用优化功能
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Optimization features will be disabled.")


@dataclass
class BacktestStats:
    """
    保存一次回测的关键指标。
    增强版：区分 Net（手续费后）和 Gross（手续费前）指标。
    """
    # Net (手续费后) 指标
    total_return: float              # 总收益
    annual_return: float             # 年化收益 (net)
    annual_vol: float                # 年化波动率 (net)
    sharpe: float                    # Sharpe 比率 (net)
    max_drawdown: float             # 最大回撤（负数，比如 -0.35）
    avg_turnover: float             # 平均每日换手率
    n_days: int                      # 回测交易天数
    
    # Gross (手续费前) 指标 - 新增
    annual_return_gross: float = 0.0
    annual_vol_gross: float = 0.0
    sharpe_gross: float = 0.0
    max_drawdown_gross: float = 0.0
    
    # 其他指标 - 新增
    margin: float = 0.0              # 保证金占用（归一化后，0-1之间）
    calmar_ratio: float = 0.0        # Calmar 比率 = 年化收益 / |最大回撤|
    sortino_ratio: float = 0.0       # Sortino 比率（下行波动率）
    win_rate: float = 0.0            # 胜率
    profit_loss_ratio: float = 0.0   # 盈亏比
    max_consecutive_losses: int = 0  # 最大连续亏损天数
    max_consecutive_wins: int = 0    # 最大连续盈利天数


@dataclass
class BacktestResult:
    """
    回测结果：包括指标和时间序列。
    """
    stats: BacktestStats
    equity_curve: pd.Series          # 权益曲线（从 1 开始）
    daily_returns: pd.Series         # 每日净收益
    turnover: pd.Series              # 每日换手率
    gross_returns: pd.Series         # 手续费前收益
    fees: pd.Series                  # 每日手续费
    position: pd.Series              # 每日仓位（新增）
    funding_fees: Optional[pd.Series] = None  # 资金费率（可选，大宗商品可能不需要）


@dataclass
class BatchBacktestResult:
    """
    批量回测结果。
    """
    results: List[BacktestResult]
    names: List[str]
    summary_df: pd.DataFrame         # 汇总对比表


class SimpleBacktester:
    """
    一个增强的本地回测引擎，不依赖任何 API。
    
    适用场景：
    - 单资产（例如 LME Lead 价格）的多空策略
    - 输入价格序列 + 预测的仓位信号（[-1, 1] 区间）
    
    基本假设：
    - 每日收盘时根据信号调整仓位
    - 当日收盘信号在下一日开盘到收盘期间生效（避免未来函数）
    - 手续费按 |Δposition| * fee_rate 来计（线性交易成本）
    """

    def __init__(
        self,
        prices: pd.Series,
        trading_days: int = 252,
        fee_rate: float = 0.0,
        funding_rate: Optional[pd.Series] = None,
        name: str = "strategy",
        risk_free_rate: float = 0.0,
    ):
        """
        参数：
            prices: 价格时间序列（index 为日期，values 为价格）
            trading_days: 一年中的交易天数（用于年化，默认 252）
            fee_rate: 单位换手成本，比如 0.0005 表示单边 5bp
            funding_rate: 资金费率时间序列（可选，用于期货等）
            name: 策略名称，仅用于画图和打印
            risk_free_rate: 无风险利率（用于计算 Sharpe 等指标）
        """
        if not isinstance(prices, pd.Series):
            raise TypeError("prices 必须是 pandas.Series")
        if prices.isnull().any():
            # 可以根据需要改成 dropna
            raise ValueError("prices 中包含 NaN，请先清洗数据")

        self.prices = prices.sort_index()
        self.trading_days = trading_days
        self.fee_rate = float(fee_rate)
        self.funding_rate = funding_rate
        self.name = name
        self.risk_free_rate = risk_free_rate

        # 预先计算日收益率
        self.returns = self.prices.pct_change().fillna(0.0)

    def run(
        self, 
        signal: pd.Series,
        max_position: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> BacktestResult:
        """
        运行回测。
        
        参数：
            signal: 仓位信号，范围通常在 [-1, 1] 之间。
                    index 必须与 prices 对齐（日期相同）。
                    signal[t] 表示当日收盘后调仓后的目标仓位，
                    实际用于下一日收益（自动做 1 天滞后）。
            max_position: 最大仓位限制（默认 1.0，即 100%）
            stop_loss: 止损比例（例如 0.1 表示 10% 止损）
            take_profit: 止盈比例（例如 0.2 表示 20% 止盈）
        
        返回：
            BacktestResult 实例，包含指标和曲线。
        """
        if not isinstance(signal, pd.Series):
            raise TypeError("signal 必须是 pandas.Series")
        if not signal.index.equals(self.prices.index):
            raise ValueError("signal 的 index 必须与 prices 完全一致（同一组日期）")

        # 限制仓位在 [-max_position, max_position] 范围内
        signal_clipped = signal.clip(-max_position, max_position)

        # 为了避免未来函数：
        # 当日的仓位用于"下一日"的收益，故向后 shift 一天
        position = signal_clipped.shift(1).fillna(0.0)

        # 应用止损/止盈逻辑
        if stop_loss is not None or take_profit is not None:
            position = self._apply_stop_loss_take_profit(
                position, stop_loss, take_profit
            )

        # 计算每日仓位变动，|Δposition|
        position_change = position.diff().fillna(position)  # 第一日变动 = position[0] - 0
        turnover = position_change.abs()

        # 手续费：|Δposition| * fee_rate
        fees = turnover * self.fee_rate

        # 资金费率（如果有）
        funding_fees = None
        if self.funding_rate is not None:
            # 资金费率通常按持仓方向计算：position * funding_rate
            funding_fees = position * self.funding_rate.reindex(
                position.index, method='ffill'
            ).fillna(0.0)
            fees = fees + funding_fees

        # 毛收益（不含手续费）
        gross_returns = position * self.returns

        # 净收益：毛收益 - 手续费
        net_returns = gross_returns - fees

        # 权益曲线，从 1 开始
        equity_curve = (1.0 + net_returns).cumprod()
        equity_curve_gross = (1.0 + gross_returns).cumprod()

        # 计算统计量
        stats = self._calculate_stats(
            net_returns, gross_returns, equity_curve, 
            equity_curve_gross, turnover, position
        )

        return BacktestResult(
            stats=stats,
            equity_curve=equity_curve,
            daily_returns=net_returns,
            turnover=turnover,
            gross_returns=gross_returns,
            fees=fees,
            position=position,
            funding_fees=funding_fees,
        )

    def _apply_stop_loss_take_profit(
        self,
        position: pd.Series,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> pd.Series:
        """
        应用止损/止盈逻辑。
        简化版本：基于权益曲线的回撤和涨幅。
        """
        # 这里实现一个简化版本
        # 实际应用中可能需要更复杂的逻辑（基于持仓成本价等）
        equity = (1.0 + (position * self.returns)).cumprod()
        rolling_max = equity.cummax()
        rolling_min = equity.cummin()
        
        drawdown = equity / rolling_max - 1.0
        runup = equity / rolling_min - 1.0
        
        adjusted_position = position.copy()
        
        if stop_loss is not None:
            # 如果回撤超过止损，平仓
            adjusted_position[drawdown < -abs(stop_loss)] = 0.0
        
        if take_profit is not None:
            # 如果涨幅超过止盈，平仓
            adjusted_position[runup > abs(take_profit)] = 0.0
        
        return adjusted_position

    def _calculate_stats(
        self,
        net_returns: pd.Series,
        gross_returns: pd.Series,
        equity_curve: pd.Series,
        equity_curve_gross: pd.Series,
        turnover: pd.Series,
        position: pd.Series,
    ) -> BacktestStats:
        """计算所有统计指标"""
        n_days = len(net_returns)
        total_return = equity_curve.iloc[-1] - 1.0
        total_return_gross = equity_curve_gross.iloc[-1] - 1.0

        # 年化收益率
        if n_days > 0:
            annual_return = (1.0 + total_return) ** (self.trading_days / n_days) - 1.0
            annual_return_gross = (1.0 + total_return_gross) ** (self.trading_days / n_days) - 1.0
        else:
            annual_return = 0.0
            annual_return_gross = 0.0

        # 年化波动率
        daily_vol = net_returns.std()
        annual_vol = daily_vol * np.sqrt(self.trading_days)
        daily_vol_gross = gross_returns.std()
        annual_vol_gross = daily_vol_gross * np.sqrt(self.trading_days)

        # Sharpe 比率
        if annual_vol > 0:
            sharpe = (annual_return - self.risk_free_rate) / annual_vol
        else:
            sharpe = 0.0
        
        if annual_vol_gross > 0:
            sharpe_gross = (annual_return_gross - self.risk_free_rate) / annual_vol_gross
        else:
            sharpe_gross = 0.0

        # 最大回撤
        rolling_max = equity_curve.cummax()
        drawdown = equity_curve / rolling_max - 1.0
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
        
        rolling_max_gross = equity_curve_gross.cummax()
        drawdown_gross = equity_curve_gross / rolling_max_gross - 1.0
        max_drawdown_gross = drawdown_gross.min() if len(drawdown_gross) > 0 else 0.0

        # Calmar 比率
        if abs(max_drawdown) > 1e-6:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Sortino 比率（下行波动率）
        downside_returns = net_returns[net_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(self.trading_days)
            if downside_vol > 0:
                sortino_ratio = (annual_return - self.risk_free_rate) / downside_vol
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0

        # 胜率和盈亏比
        winning_trades = net_returns[net_returns > 0]
        losing_trades = net_returns[net_returns < 0]
        
        if len(net_returns) > 0:
            win_rate = len(winning_trades) / len(net_returns)
        else:
            win_rate = 0.0
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = winning_trades.mean()
            avg_loss = abs(losing_trades.mean())
            if avg_loss > 0:
                profit_loss_ratio = avg_win / avg_loss
            else:
                profit_loss_ratio = 0.0
        else:
            profit_loss_ratio = 0.0

        # 最大连续亏损/盈利
        max_consecutive_losses = self._max_consecutive(net_returns < 0)
        max_consecutive_wins = self._max_consecutive(net_returns > 0)

        # 保证金占用（平均绝对仓位）
        margin = position.abs().mean()

        avg_turnover = turnover.mean()

        return BacktestStats(
            total_return=float(total_return),
            annual_return=float(annual_return),
            annual_vol=float(annual_vol),
            sharpe=float(sharpe),
            max_drawdown=float(max_drawdown),
            avg_turnover=float(avg_turnover),
            n_days=int(n_days),
            annual_return_gross=float(annual_return_gross),
            annual_vol_gross=float(annual_vol_gross),
            sharpe_gross=float(sharpe_gross),
            max_drawdown_gross=float(max_drawdown_gross),
            margin=float(margin),
            calmar_ratio=float(calmar_ratio),
            sortino_ratio=float(sortino_ratio),
            win_rate=float(win_rate),
            profit_loss_ratio=float(profit_loss_ratio),
            max_consecutive_losses=int(max_consecutive_losses),
            max_consecutive_wins=int(max_consecutive_wins),
        )

    @staticmethod
    def _max_consecutive(condition: pd.Series) -> int:
        """计算最大连续满足条件的次数"""
        if len(condition) == 0:
            return 0
        groups = (condition != condition.shift()).cumsum()
        return condition.groupby(groups).sum().max() if condition.any() else 0

    @staticmethod
    def batch_run(
        backtesters: List['SimpleBacktester'],
        signals: List[pd.Series],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> BatchBacktestResult:
        """
        批量回测多个策略。
        
        参数：
            backtesters: SimpleBacktester 实例列表
            signals: 对应的信号列表
            names: 策略名称列表（可选）
            **kwargs: 传递给 run() 的其他参数
        
        返回：
            BatchBacktestResult
        """
        if len(backtesters) != len(signals):
            raise ValueError("backtesters 和 signals 长度必须一致")
        
        if names is None:
            names = [bt.name for bt in backtesters]
        
        results = []
        for bt, sig, name in zip(backtesters, signals, names):
            bt.name = name  # 更新名称
            result = bt.run(sig, **kwargs)
            results.append(result)
        
        # 生成对比汇总表
        summary_data = []
        for result, name in zip(results, names):
            stats = result.stats
            summary_data.append({
                'Strategy': name,
                'Total Return': stats.total_return,
                'Annual Return (Net)': stats.annual_return,
                'Annual Return (Gross)': stats.annual_return_gross,
                'Sharpe (Net)': stats.sharpe,
                'Sharpe (Gross)': stats.sharpe_gross,
                'Max Drawdown (Net)': stats.max_drawdown,
                'Max Drawdown (Gross)': stats.max_drawdown_gross,
                'Calmar Ratio': stats.calmar_ratio,
                'Sortino Ratio': stats.sortino_ratio,
                'Win Rate': stats.win_rate,
                'Avg Turnover': stats.avg_turnover,
                'Margin': stats.margin,
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        return BatchBacktestResult(
            results=results,
            names=names,
            summary_df=summary_df
        )

    def optimize_parameters(
        self,
        signal_func: Callable[[pd.Series, Dict[str, Any]], pd.Series],
        param_ranges: Dict[str, Union[List[Any], Tuple[float, float]]],
        optimization_metric: str = "sharpe",
        optimization_direction: str = "maximize",
        n_trials: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Optuna 优化策略参数。
        
        参数：
            signal_func: 信号生成函数，接受 (prices, params) 返回 signal Series
            param_ranges: 参数范围，例如 {"window": [5, 10, 20], "threshold": (0.1, 0.5)}
            optimization_metric: 优化目标（"sharpe", "annual_return", "calmar_ratio" 等）
            optimization_direction: "maximize" 或 "minimize"
            n_trials: 优化试验次数
            **kwargs: 传递给 run() 的其他参数
        
        返回：
            包含最佳参数和结果的字典
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for parameter optimization. Install with: pip install optuna")

        def objective(trial):
            # 根据 param_ranges 生成参数
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, tuple):
                    # 连续范围
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # 离散值
                    if all(isinstance(x, (int, float)) for x in param_range):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    raise ValueError(f"Unsupported param_range type for {param_name}")

            # 生成信号
            signal = signal_func(self.prices, params)
            
            # 运行回测
            result = self.run(signal, **kwargs)
            
            # 获取优化指标
            stats = result.stats
            metric_value = getattr(stats, optimization_metric, stats.sharpe)
            
            return metric_value

        study = optuna.create_study(
            direction=optimization_direction,
            study_name=f"{self.name}_optimization"
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 使用最佳参数运行最终回测
        best_params = study.best_params
        best_signal = signal_func(self.prices, best_params)
        best_result = self.run(best_signal, **kwargs)
        
        return {
            'best_params': best_params,
            'best_value': study.best_value,
            'best_result': best_result,
            'study': study,
            'optimization_metric': optimization_metric,
        }

    @staticmethod
    def summarize(result: BacktestResult) -> Dict[str, Any]:
        """
        把结果整理成一个字典，方便打印或转成 DataFrame。
        """
        s = result.stats
        return {
            "Total Return": s.total_return,
            "Annual Return (Net)": s.annual_return,
            "Annual Return (Gross)": s.annual_return_gross,
            "Annual Vol (Net)": s.annual_vol,
            "Annual Vol (Gross)": s.annual_vol_gross,
            "Sharpe (Net)": s.sharpe,
            "Sharpe (Gross)": s.sharpe_gross,
            "Max Drawdown (Net)": s.max_drawdown,
            "Max Drawdown (Gross)": s.max_drawdown_gross,
            "Calmar Ratio": s.calmar_ratio,
            "Sortino Ratio": s.sortino_ratio,
            "Win Rate": s.win_rate,
            "Profit/Loss Ratio": s.profit_loss_ratio,
            "Max Consecutive Losses": s.max_consecutive_losses,
            "Max Consecutive Wins": s.max_consecutive_wins,
            "Avg Turnover": s.avg_turnover,
            "Margin": s.margin,
            "N Days": s.n_days,
        }

    @staticmethod
    def display_results(result: Union[BacktestResult, BatchBacktestResult]):
        """
        格式化显示回测结果（类似 Casimir 的 display_results）。
        """
        if isinstance(result, BacktestResult):
            stats = result.stats
            print("\n" + "=" * 60)
            print(f"回测结果: {result.stats}")
            print("=" * 60)
            print(f"\n【核心指标】")
            print(f"  总收益: {stats.total_return:.2%}")
            print(f"  年化收益 (Net/Gross): {stats.annual_return:.2%} / {stats.annual_return_gross:.2%}")
            print(f"  年化波动率 (Net/Gross): {stats.annual_vol:.2%} / {stats.annual_vol_gross:.2%}")
            print(f"  Sharpe 比率 (Net/Gross): {stats.sharpe:.3f} / {stats.sharpe_gross:.3f}")
            print(f"  最大回撤 (Net/Gross): {stats.max_drawdown:.2%} / {stats.max_drawdown_gross:.2%}")
            print(f"\n【风险指标】")
            print(f"  Calmar 比率: {stats.calmar_ratio:.3f}")
            print(f"  Sortino 比率: {stats.sortino_ratio:.3f}")
            print(f"  最大连续亏损: {stats.max_consecutive_losses} 天")
            print(f"  最大连续盈利: {stats.max_consecutive_wins} 天")
            print(f"\n【交易统计】")
            print(f"  胜率: {stats.win_rate:.2%}")
            print(f"  盈亏比: {stats.profit_loss_ratio:.3f}")
            print(f"  平均换手率: {stats.avg_turnover:.4f}")
            print(f"  保证金占用: {stats.margin:.2%}")
            print(f"  回测天数: {stats.n_days}")
        
        elif isinstance(result, BatchBacktestResult):
            print("\n" + "=" * 60)
            print("批量回测结果对比")
            print("=" * 60)
            print("\n" + result.summary_df.to_string(index=False))

    @staticmethod
    def plot_equity(
        result: BacktestResult, 
        title: Optional[str] = None,
        show_gross: bool = True,
        save_path: Optional[str] = None
    ):
        """
        画权益曲线和回撤。
        增强版：可以同时显示 net 和 gross 曲线。
        """
        equity = result.equity_curve
        equity_gross = (1.0 + result.gross_returns).cumprod()
        dd = equity / equity.cummax() - 1.0
        dd_gross = equity_gross / equity_gross.cummax() - 1.0

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 权益曲线
        axes[0].plot(equity.index, equity.values, label="Equity (Net)", linewidth=1.5)
        if show_gross:
            axes[0].plot(equity_gross.index, equity_gross.values, 
                        label="Equity (Gross)", linewidth=1.5, alpha=0.7, linestyle='--')
        axes[0].set_ylabel("Equity")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[0].set_title("权益曲线")

        # 回撤
        axes[1].fill_between(dd.index, dd.values, 0, label="Drawdown (Net)", 
                            color="red", alpha=0.3)
        if show_gross:
            axes[1].plot(dd_gross.index, dd_gross.values, 
                        label="Drawdown (Gross)", color="orange", linewidth=1, alpha=0.7)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        axes[1].set_title("回撤")

        # 仓位
        axes[2].plot(result.position.index, result.position.values, 
                    label="Position", color="green", linewidth=1, alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_ylabel("Position")
        axes[2].set_xlabel("Date")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
        axes[2].set_title("仓位变化")

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle("回测结果", fontsize=14, fontweight='bold')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()

    @staticmethod
    def plot_comparison(batch_result: BatchBacktestResult, save_path: Optional[str] = None):
        """
        绘制批量回测结果的对比图。
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 权益曲线对比
        for result, name in zip(batch_result.results, batch_result.names):
            axes[0].plot(result.equity_curve.index, result.equity_curve.values, 
                        label=name, linewidth=1.5)
        
        axes[0].set_ylabel("Equity")
        axes[0].set_title("策略对比 - 权益曲线")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        # 回撤对比
        for result, name in zip(batch_result.results, batch_result.names):
            dd = result.equity_curve / result.equity_curve.cummax() - 1.0
            axes[1].plot(dd.index, dd.values, label=name, linewidth=1.5)
        
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].set_title("策略对比 - 回撤")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存至: {save_path}")
        
        plt.show()