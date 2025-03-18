import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """加载并处理纳斯达克100的数据"""
    df = pd.read_csv(file_path)
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df.set_index('交易日期', inplace=True)
    return df

def calculate_portfolio_returns(
    ndx_data: pd.DataFrame,
    strategy_df: pd.DataFrame
) -> Tuple[pd.Series, float]:
    """计算投资组合的收益和beta"""
    # 确保权重数据有正确的索引
    strategy_df = strategy_df.copy()
    strategy_df['date'] = pd.to_datetime(strategy_df['date'])
    strategy_df = strategy_df.set_index('date')
    
    # 确保索引对齐
    common_dates = ndx_data.index.intersection(strategy_df.index)
    ndx_data = ndx_data.loc[common_dates]
    strategy_df = strategy_df.loc[common_dates]
    
    ndx_weights = strategy_df['ndx_weight']
    bond_weights = strategy_df['bond_weight']
    
    # 计算纳斯达克100的日收益率
    ndx_returns = ndx_data['收盘价'].pct_change()
    
    # 债券的日收益率（假设年化3%）
    daily_bond_return = (1 + 0.03) ** (1/252) - 1
    
    # 计算组合收益
    portfolio_returns = ndx_returns * ndx_weights + daily_bond_return * bond_weights
    portfolio_returns = portfolio_returns.fillna(0)
    
    # 计算beta
    market_var = np.var(ndx_returns)
    portfolio_covar = np.cov(portfolio_returns, ndx_returns)[0][1]
    beta = portfolio_covar / market_var if market_var != 0 else 1
    
    return portfolio_returns, beta

def grid_trading_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """实现网格交易策略"""
    results = []
    portfolio_value = 1000000  # 初始资金100万
    ndx_weight = 0.6  # 初始配置60%股票
    last_pe = None  # 记录上一个交易日的PE
    
    for i in range(len(df)):
        pe = df.iloc[i]['市盈率TTM']
        
        if last_pe is not None:  # 不是第一个交易日
            if last_pe <= 32:  # 如果上一个交易日PE <= 32，只能买入不能卖出
                # 计算当前可能的目标权重
                target_weight = ndx_weight  # 默认保持不变
                if pe <= 22:
                    target_weight = max(ndx_weight, 1.0)
                elif pe <= 23:
                    target_weight = max(ndx_weight, 0.95)
                elif pe <= 25:
                    target_weight = max(ndx_weight, 0.90)
                elif pe <= 27:
                    target_weight = max(ndx_weight, 0.80)
                elif pe <= 29:
                    target_weight = max(ndx_weight, 0.70)
                elif pe <= 32:
                    target_weight = max(ndx_weight, 0.60)
                ndx_weight = target_weight
            else:  # 如果上一个交易日PE > 32，执行减仓操作
                if pe > 37:
                    ndx_weight = 0.60
                elif pe > 35:
                    ndx_weight = 0.70
                elif pe > 33:
                    ndx_weight = 0.80
                elif pe > 32:
                    ndx_weight = 0.90
        
        last_pe = pe  # 更新上一个交易日的PE
        bond_weight = 1 - ndx_weight
        results.append({
            'date': df.index[i],
            'ndx_weight': ndx_weight,
            'bond_weight': bond_weight
        })
    
    return pd.DataFrame(results)

def balanced_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """实现股债平衡策略"""
    results = []
    for i in range(len(df)):
        date = df.index[i]
        if i == 0 or date.month != df.index[i-1].month:
            results.append({
                'date': date,
                'ndx_weight': 0.5,
                'bond_weight': 0.5
            })
        else:
            results.append(results[-1].copy())
    
    return pd.DataFrame(results)

def buy_and_hold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """实现买入持有策略"""
    return pd.DataFrame({
        'date': df.index,
        'ndx_weight': [1.0] * len(df),
        'bond_weight': [0.0] * len(df)
    })

def generate_report(df: pd.DataFrame) -> None:
    """生成回测报告"""
    # 实现三种策略
    grid_results = grid_trading_strategy(df)
    balanced_results = balanced_strategy(df)
    hold_results = buy_and_hold_strategy(df)
    
    # 计算每个策略的月度统计数据
    strategies = {
        '网格交易': grid_results,
        '股债平衡': balanced_results,
        '买入持有': hold_results
    }
    
    monthly_stats = []
    
    for name, strategy_df in strategies.items():
        returns, beta = calculate_portfolio_returns(df, strategy_df)
        
        # 计算月度统计
        monthly_returns = returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        ndx_monthly = df['收盘价'].resample('M').last().pct_change()
        
        # 计算累计净值
        cumulative_value = (1 + monthly_returns).cumprod()
        
        for date, ret in monthly_returns.items():
            monthly_stats.append({
                '日期': date,
                '策略': name,
                '组合收益率': ret * 100,
                '纳斯达克收益率': ndx_monthly[date] * 100 if date in ndx_monthly.index else 0,
                'Beta': beta,
                '净值': cumulative_value[date]
            })
    
    # 生成报告
    report_df = pd.DataFrame(monthly_stats)
    report_df.to_markdown('./report.md', index=False)

if __name__ == "__main__":
    # 加载数据
    ndx_data = load_and_process_data('./NDX.csv')
    
    # 生成报告
    generate_report(ndx_data)
