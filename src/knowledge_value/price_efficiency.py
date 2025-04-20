#!/usr/bin/env python3
'''
Price Efficiency

This module will contain code for analyzing Polymarket data focusing on price efficiency.
'''

def test_price_efficiency(market_id, trades_dir='data/trades'):
    """
    Test for price efficiency using random walk hypothesis
    """
    trades_df = load_trade_data(market_id, trades_dir)
    if trades_df is None or len(trades_df) < 30:
        print(f"Insufficient trade data for market {market_id}")
        return None
    
    # Calculate daily price series
    trades_df['date'] = trades_df['timestamp'].dt.date
    
    # Calculate VWAP (Volume-Weighted Average Price) for each day
    trades_df['volume'] = trades_df['makerAmountFilled'] + trades_df['takerAmountFilled']
    trades_df['price_volume'] = trades_df['price'] * trades_df['volume']
    
    daily_prices = trades_df.groupby('date').apply(
        lambda x: x['price_volume'].sum() / x['volume'].sum()
    ).reset_index()
    daily_prices.columns = ['date', 'price']
    
    # Calculate returns
    daily_prices['return'] = daily_prices['price'].pct_change()
    daily_prices = daily_prices.dropna()
    
    if len(daily_prices) < 5:
        return None
    
    # Test 1: Autocorrelation of returns (lag 1)
    from scipy.stats import pearsonr
    
    lag1_returns = daily_prices['return'].shift(1).dropna()
    current_returns = daily_prices['return'].iloc[1:]
    
    corr, p_value = pearsonr(lag1_returns, current_returns)
    
    # Test 2: Runs test for randomness
    from statsmodels.stats.diagnostic import runs_test
    
    median = daily_prices['return'].median()
    runs, p_value_runs = runs_test(daily_prices['return'], median)
    
    results = {
        'market_id': market_id,
        'n_days': len(daily_prices),
        'autocorrelation': corr,
        'autocorr_p_value': p_value,
        'runs_statistic': runs,
        'runs_p_value': p_value_runs,
        'is_efficient': (p_value > 0.05) and (p_value_runs > 0.05)
    }
    
    return results, daily_prices