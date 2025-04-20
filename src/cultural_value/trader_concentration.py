#!/usr/bin/env python3
'''
Trader Concentration

This module will contain code for analyzing Polymarket data focusing on trader concentration.
'''

def calculate_trader_concentration(market_id, trades_dir='data/trades'):
    """
    Calculate trader concentration metrics for a market
    """
    trades_df = load_trade_data(market_id, trades_dir)
    if trades_df is None:
        print(f"No trade data found for market {market_id}")
        return None
    
    # Identify unique traders
    makers = set(trades_df['maker'].unique())
    takers = set(trades_df['taker'].unique())
    all_traders = makers.union(takers)
    
    # Calculate volume by trader
    trades_df['total_volume'] = trades_df['makerAmountFilled'] + trades_df['takerAmountFilled']
    
    # Volume by maker
    maker_volume = trades_df.groupby('maker')['makerAmountFilled'].sum()
    
    # Volume by taker
    taker_volume = trades_df.groupby('taker')['takerAmountFilled'].sum()
    
    # Combine volumes
    trader_volumes = pd.DataFrame({
        'maker_volume': maker_volume,
        'taker_volume': taker_volume
    }).fillna(0)
    
    trader_volumes['total_volume'] = trader_volumes['maker_volume'] + trader_volumes['taker_volume']
    trader_volumes = trader_volumes.sort_values('total_volume', ascending=False)
    
    # Calculate concentration metrics
    total_volume = trader_volumes['total_volume'].sum()
    trader_volumes['volume_share'] = trader_volumes['total_volume'] / total_volume
    trader_volumes['cumulative_share'] = trader_volumes['volume_share'].cumsum()
    
    # Calculate Gini coefficient
    from scipy.stats import gini
    gini_coef = gini(trader_volumes['total_volume'])
    
    # Calculate share of top traders
    top_1pct_count = max(1, int(len(trader_volumes) * 0.01))
    top_5pct_count = max(1, int(len(trader_volumes) * 0.05))
    top_10pct_count = max(1, int(len(trader_volumes) * 0.10))
    
    top_1pct_share = trader_volumes['volume_share'].iloc[:top_1pct_count].sum()
    top_5pct_share = trader_volumes['volume_share'].iloc[:top_5pct_count].sum()
    top_10pct_share = trader_volumes['volume_share'].iloc[:top_10pct_count].sum()
    
    results = {
        'market_id': market_id,
        'unique_traders': len(all_traders),
        'gini_coefficient': gini_coef,
        'top_1pct_share': top_1pct_share,
        'top_5pct_share': top_5pct_share,
        'top_10pct_share': top_10pct_share,
        'hhi_index': (trader_volumes['volume_share'] ** 2).sum()  # Herfindahl-Hirschman Index
    }
    
    return results, trader_volumes

def plot_trader_concentration(trader_volumes, market_id=None, market_question_map=None, save_path=None):
    """
    Plot Lorenz curve and concentration metrics for a market
    """
    # Sort by volume share for Lorenz curve
    trader_volumes = trader_volumes.sort_values('volume_share')
    trader_volumes['lorenz_curve'] = trader_volumes['volume_share'].cumsum()
    trader_volumes['perfect_equality'] = np.linspace(0, 1, len(trader_volumes))
    
    # Get market title if available
    market_title = f"Market ID: {market_id}"
    if market_question_map and market_id in market_question_map:
        market_title = market_question_map[market_id]
    
    from scipy.stats import gini
    gini_coef = gini(trader_volumes['total_volume'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Lorenz curve
    ax.plot(np.linspace(0, 1, len(trader_volumes)), trader_volumes['lorenz_curve'], 
            label='Lorenz Curve', color='#E91E63')
    
    # Plot line of perfect equality
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
    
    # Fill area between curves
    ax.fill_between(np.linspace(0, 1, len(trader_volumes)), 
                   trader_volumes['perfect_equality'], 
                   trader_volumes['lorenz_curve'], 
                   alpha=0.2, color='#E91E63')
    
    ax.set_xlabel('Cumulative Share of Traders')
    ax.set_ylabel('Cumulative Share of Volume')
    ax.set_title(f"Trader Concentration: {market_title}\nGini Coefficient: {gini_coef:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()