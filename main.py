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

# 计算 beta
def calculate_beta(rm_data: pd.Series, re_data: pd.Series) -> pd.Series:
    # 市场的日收益率
    rm_returns = rm_data.pct_change().fillna(0)
    # 组合的收益率
    re_returns = re_data.pct_change().fillna(0)
    # 计算纳斯达克100的beta系数
    beta = rm_returns.cov(re_returns) / rm_returns.var()
    return beta

def calculate_portfolio_values(
    initial_portfolio_value: float,
    ndx_data: pd.DataFrame,
    strategy_df: pd.DataFrame
) -> pd.Series:
    """计算投资组合的净值"""
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
    ndx_returns = ndx_data['收盘价'].pct_change().fillna(0)
    
    # 债券的日收益率（假设年化3%）
    # daily_bond_return = pd.Series((1 + 0.03) ** (1/252) - 1, index=bond_weights.index)
    daily_bond_return = pd.Series(0, index=bond_weights.index)
    
    # 计算组合净值
    portfolio_values = pd.Series(index=common_dates, dtype=float)
    portfolio_values.iloc[0] = initial_portfolio_value
    
    for i in range(1, len(common_dates)):
        date = common_dates[i]
        prev_value = portfolio_values.iloc[i-1]
        ndx_return = ndx_returns.iloc[i]
        bond_return = daily_bond_return.iloc[i]
        
        # 计算当日净值
        portfolio_values.iloc[i] = prev_value * (1.0 + ndx_return * ndx_weights.iloc[i] + bond_return * bond_weights.iloc[i])
    
    return portfolio_values, calculate_beta(ndx_data['收盘价'], portfolio_values)

def grid_trading_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """实现网格交易策略"""
    results = []
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
                else:
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
                else:
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
        if i == 0 or date.year != df.index[i-1].year:
            results.append({
                'date': date,
                'ndx_weight': 0.5,
                'bond_weight': 0.5
            })
        else:
            copy = results[-1].copy()
            copy['date'] = date
            results.append(copy)
            
            # 计算纳斯达克的涨跌幅
            ndx_return = df.iloc[i]['收盘价'] / df.iloc[i-1]['收盘价'] - 1
            
            # 根据纳斯达克的涨跌幅调整比例
            ndx_weight = results[-1]['ndx_weight'] * (1 + ndx_return)
            bond_weight = 1 - ndx_weight  # 保持总比例为1
            
            results.append({
                'date': date,
                'ndx_weight': ndx_weight,
                'bond_weight': bond_weight
            })
    
    return pd.DataFrame(results)

def buy_and_hold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """实现买入持有策略"""
    return pd.DataFrame({
        'date': df.index,
        'ndx_weight': [1.0] * len(df),
        'bond_weight': [0.0] * len(df)
    })

def generate_report(df: pd.DataFrame, initial_portfolio_value: float) -> None:
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
        returns, beta = calculate_portfolio_values(initial_portfolio_value, df, strategy_df)
        # 确保数据有正确的索引
        strategy_df_copy = strategy_df.copy()
        if not isinstance(strategy_df_copy.index, pd.DatetimeIndex):
            strategy_df_copy['date'] = pd.to_datetime(strategy_df_copy['date'])
            strategy_df_copy.set_index('date', inplace=True)
        strategy_df_copy = strategy_df_copy.resample('M').last().ffill()
        
        # 计算每月最后一个交易日的净值
        monthly_end_values = returns.resample('M').last().ffill()
        
        # 计算月度收益率：当月最后一个交易日与上月最后一个交易日的对比
        monthly_returns = monthly_end_values.pct_change()
        
        ndx_monthly = df.resample('M').last().ffill()
        ndx_monthly_pct = df['收盘价'].resample('M').last().ffill().pct_change()
        
        # 计算组合累计收益率
        for date, ret in monthly_returns.items():
            # 检查 strategy_df 是否包含当前日期
            ndx_weight = strategy_df_copy.loc[date, 'ndx_weight']
                
            monthly_stats.append({
                '日期': date,
                '策略': f'{name}-{ndx_weight}',
                'pe': ndx_monthly.loc[date, '市盈率TTM'],
                '组合当月收益率': f"{ret * 100:.2f}%",  # 将变化率转换为百分比
                # '组合当月净值': f"{monthly_end_values[date]:.2f}",
                '纳斯达克当月收益率': f"{ndx_monthly_pct[date] * 100 if date in ndx_monthly_pct.index else 0:.2f}%",
                # '纳斯达克当月净值': f"{ndx_monthly.loc[date, '收盘价']:.2f}",
                # '每月差值': f"{(ndx_monthly_pct[date] * 100 if date in ndx_monthly_pct.index else 0) * ndx_weight - ret * 100:.2f}%",
                'beta': beta,
                '组合累计收益率': f"{(monthly_end_values[date] / initial_portfolio_value - 1) * 100:.2f}%",
                '纳斯达克累计收益率': f"{(ndx_monthly.loc[date, '收盘价']/df['收盘价'][0] - 1) * 100:.2f}%",
            })
    
        # 生成报告
        report_df = pd.DataFrame(monthly_stats)
        # 使用策略名称动态生成文件名
        report_filename = f'./report-{name}.md'
        report_df.to_markdown(report_filename, index=False)
        monthly_stats.clear()

if __name__ == "__main__":
    # 加载数据
    ndx_data = load_and_process_data('./NDX.csv')
    
    # 生成报告
    generate_report(ndx_data, 100000)