# src/trader_analysis.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Make sure we can import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import data utilities with fallback options
try:
    from utils.data_loader import load_trade_data, get_sample_market_ids, load_market_question_mapping
except ImportError:
    try:
        from src.utils.data_loader import load_trade_data, get_sample_market_ids, load_market_question_mapping
    except ImportError:
        print("Error importing data_loader utilities. Please run this script from the project root directory.")
        sys.exit(1)

def load_sample_data(market_ids=None, n_markets=3):
    """
    Load sample market data with fallback options
    
    Returns:
        Tuple of (market_ids, market_questions)
    """
    # Try to get market IDs
    if market_ids is None:
        try:
            market_ids = get_sample_market_ids(n_markets)
        except Exception as e:
            print(f"Warning: Could not get sample market IDs: {e}")
            # Fallback to some sample IDs
            market_ids = [f"market_{i}" for i in range(1, n_markets+1)]
    
    # Try to get market questions mapping
    try:
        market_questions = load_market_question_mapping()
    except Exception as e:
        print(f"Warning: Could not load market questions: {e}")
        # Create empty mapping
        market_questions = {}
    
    return market_ids, market_questions

def analyze_traders(market_ids=None, n_markets=3, min_trades=5, save_path='results/trader_analysis'):
    """
    Run comprehensive trader analysis on specified markets
    
    Args:
        market_ids: List of market IDs to analyze (if None, samples n_markets)
        n_markets: Number of markets to sample if market_ids not provided
        min_trades: Minimum trades for a trader to be included
        save_path: Directory to save results
    
    Returns:
        Dictionary with analysis results
    """
    # Setup
    market_ids, market_questions = load_sample_data(market_ids, n_markets)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Analyzing {len(market_ids)} markets...")
    
    # Process each market
    all_traders_data = []
    market_summaries = []
    
    for market_id in market_ids:
        market_name = market_questions.get(str(market_id), f"Market {market_id}")
        print(f"\nProcessing market: {market_name}")
        
        # Load trades
        try:
            trades = load_trade_data(market_id)
            if trades is None or len(trades) < 100:
                print(f"  Insufficient data, skipping")
                continue
                
            # Ensure numeric columns
            if 'size' in trades.columns:
                trades['volume'] = pd.to_numeric(trades['size'], errors='coerce')
            elif 'volume' not in trades.columns:
                print("  Warning: No size/volume column found")
                trades['volume'] = 1.0  # Default to 1.0 for missing volume
                
            if 'price' in trades.columns:
                trades['price'] = pd.to_numeric(trades['price'], errors='coerce')
            elif 'price_num' in trades.columns:
                trades['price'] = pd.to_numeric(trades['price_num'], errors='coerce')
            else:
                print("  Warning: No price column found")
                trades['price'] = 0.5  # Default to 0.5 for missing price
                
        except Exception as e:
            print(f"  Error loading trade data: {e}")
            continue
            
        # Get all traders
        makers = set(trades['maker_id'].unique())
        takers = set(trades['taker_id'].unique())
        all_traders = makers.union(takers)
        
        print(f"  Trades: {len(trades)}, Traders: {len(all_traders)}")
        
        # Extract trader behaviors
        trader_stats = {}
        
        for trader_id in all_traders:
            # Get all trades for this trader
            maker_trades = trades[trades['maker_id'] == trader_id]
            taker_trades = trades[trades['taker_id'] == trader_id]
            total_trades = len(maker_trades) + len(taker_trades)
            
            if total_trades < min_trades:
                continue
                
            # Calculate basic metrics
            maker_volume = maker_trades['volume'].sum() if not maker_trades.empty else 0
            taker_volume = taker_trades['volume'].sum() if not taker_trades.empty else 0
            total_volume = maker_volume + taker_volume
            
            # Calculate P&L (simplified)
            # Handle both 'side' column formats: 'Buy'/'Sell' or 0/1
            buy_side_values = ['Buy', 1]
            sell_side_values = ['Sell', 0]
            
            # Process buys (when trader is buying)
            buys_as_taker = taker_trades[taker_trades['side'].isin(buy_side_values)] if 'side' in taker_trades.columns else pd.DataFrame()
            buys_as_maker = maker_trades[maker_trades['side'].isin(sell_side_values)] if 'side' in maker_trades.columns else pd.DataFrame()
            buys = pd.concat([buys_as_maker, buys_as_taker])
            
            # Process sells (when trader is selling)
            sells_as_taker = taker_trades[taker_trades['side'].isin(sell_side_values)] if 'side' in taker_trades.columns else pd.DataFrame()
            sells_as_maker = maker_trades[maker_trades['side'].isin(buy_side_values)] if 'side' in maker_trades.columns else pd.DataFrame()
            sells = pd.concat([sells_as_maker, sells_as_taker])
            
            buy_volume = buys['volume'].sum()
            buy_cost = (buys['volume'] * buys['price']).sum()
            
            sell_volume = sells['volume'].sum()
            sell_value = (sells['volume'] * sells['price']).sum()
            
            # Simple PnL estimate
            pnl = sell_value - buy_cost
            
            # Store trader data
            trader_stats[trader_id] = {
                'trader_id': trader_id,
                'market_id': market_id,
                'total_trades': total_trades,
                'maker_ratio': len(maker_trades) / total_trades if total_trades > 0 else 0,
                'total_volume': total_volume,
                'avg_trade_size': total_volume / total_trades if total_trades > 0 else 0,
                'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else 10.0,
                'pnl': pnl
            }
        
        # Convert to DataFrame
        market_traders = pd.DataFrame.from_dict(trader_stats, orient='index')
        
        if len(market_traders) == 0:
            print(f"  No traders with at least {min_trades} trades, skipping")
            continue
            
        # Add to combined dataset
        all_traders_data.append(market_traders)
        
        # Calculate market-level metrics
        
        # 1. Trader concentration (Gini coefficient)
        gini_trades = calculate_gini(market_traders['total_trades'])
        gini_volume = calculate_gini(market_traders['total_volume'])
        
        # 2. Volume by top traders
        sorted_by_volume = market_traders.sort_values('total_volume', ascending=False)
        total_market_volume = sorted_by_volume['total_volume'].sum()
        
        top10pct_idx = max(1, int(len(sorted_by_volume) * 0.1))
        vol_by_top10pct = sorted_by_volume.iloc[:top10pct_idx]['total_volume'].sum() / total_market_volume
        
        # 3. Profit distribution
        profitable_traders = market_traders[market_traders['pnl'] > 0]
        pct_profitable = len(profitable_traders) / len(market_traders)
        
        # 4. Profit concentration
        if len(profitable_traders) > 0:
            sorted_by_profit = profitable_traders.sort_values('pnl', ascending=False)
            total_profit = sorted_by_profit['pnl'].sum()
            
            top10pct_profit_idx = max(1, int(len(sorted_by_profit) * 0.1))
            profit_by_top10pct = sorted_by_profit.iloc[:top10pct_profit_idx]['pnl'].sum() / total_profit
        else:
            profit_by_top10pct = 0
        
        # Store market summary
        market_summaries.append({
            'market_id': market_id,
            'market_name': market_name,
            'n_traders': len(market_traders),
            'n_trades': len(trades),
            'gini_trades': gini_trades,
            'gini_volume': gini_volume,
            'vol_by_top10pct': vol_by_top10pct,
            'pct_profitable': pct_profitable,
            'profit_by_top10pct': profit_by_top10pct
        })
        
        # Create visualizations
        try:
            plt.figure(figsize=(12, 10))
            
            # Trader concentration (Lorenz curve)
            plt.subplot(2, 2, 1)
            plot_lorenz_curve(market_traders['total_volume'], 'Trading Volume')
            
            # Profit distribution
            plt.subplot(2, 2, 2)
            plt.hist(market_traders['pnl'], bins=20)
            plt.axvline(0, color='r', linestyle='--')
            plt.title('Profit Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Number of Traders')
            
            # Trading style
            plt.subplot(2, 2, 3)
            plt.scatter(market_traders['maker_ratio'], market_traders['avg_trade_size'], 
                       alpha=0.5, c=market_traders['pnl'], cmap='coolwarm')
            plt.colorbar(label='PnL')
            plt.title('Trading Style')
            plt.xlabel('Maker Ratio')
            plt.ylabel('Average Trade Size')
            
            # Volume vs PnL
            plt.subplot(2, 2, 4)
            plt.scatter(market_traders['total_volume'], market_traders['pnl'], alpha=0.5)
            plt.axhline(0, color='r', linestyle='--')
            plt.xscale('log')
            plt.title('Volume vs PnL')
            plt.xlabel('Total Volume')
            plt.ylabel('PnL')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'trader_analysis_{market_id}.png'))
            plt.close()
        except Exception as e:
            print(f"  Error creating visualizations: {e}")
    
    # Combine all trader data
    if not all_traders_data:
        print("No valid markets found with sufficient data")
        return None
        
    combined_traders = pd.concat(all_traders_data)
    market_summary_df = pd.DataFrame(market_summaries)
    
    # Save summary data
    market_summary_df.to_csv(os.path.join(save_path, 'market_summaries.csv'), index=False)
    combined_traders.to_csv(os.path.join(save_path, 'all_traders.csv'), index=False)
    
    # Classify trader types
    trader_types = classify_traders(combined_traders)
    
    # Create summary visualizations
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='type', y='count', data=trader_types['type_summary'])
        plt.title('Trader Type Distribution')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_path, 'trader_types.png'), bbox_inches='tight')
        plt.close()
        
        # Create concentration summary
        plt.figure(figsize=(10, 6))
        plt.bar(['Trade Concentration', 'Volume Concentration', 'Profit Concentration'], 
               [market_summary_df['gini_trades'].mean(), 
                market_summary_df['gini_volume'].mean(), 
                market_summary_df['profit_by_top10pct'].mean()],
               color=['blue', 'green', 'red'])
        plt.title('Average Market Concentration Metrics')
        plt.ylabel('Concentration (Gini Coefficient / % by Top 10%)')
        plt.savefig(os.path.join(save_path, 'concentration_summary.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating summary visualizations: {e}")
    
    return {
        'market_summaries': market_summary_df,
        'trader_types': trader_types,
        'all_traders': combined_traders
    }

def calculate_gini(values):
    """Calculate Gini coefficient (0=equal, 1=unequal)"""
    # Handle edge cases
    if len(values) <= 1 or values.sum() == 0:
        return 0
        
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Calculate using cumulative distribution
    cum_values = np.cumsum(sorted_values)
    return 1 - 2 * np.sum(cum_values / cum_values[-1]) / n + 1 / n

def plot_lorenz_curve(values, title):
    """Plot Lorenz curve for inequality visualization"""
    # Handle edge cases
    if len(values) <= 1 or values.sum() == 0:
        plt.text(0.5, 0.5, "Insufficient data", ha='center')
        return
        
    sorted_values = np.sort(values)
    cumsum = np.cumsum(sorted_values)
    
    # Calculate normalized cumulative distribution
    y_lorenz = cumsum / cumsum[-1]
    x_lorenz = np.arange(1, len(values) + 1) / len(values)
    
    plt.plot(x_lorenz, y_lorenz, label='Lorenz curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect equality')
    plt.fill_between(x_lorenz, x_lorenz, y_lorenz, alpha=0.2)
    plt.title(f'Lorenz Curve - {title}')
    plt.xlabel('Cumulative % of traders')
    plt.ylabel(f'Cumulative % of {title}')
    plt.legend()

def classify_traders(trader_data, n_clusters=5):
    """
    Classify traders into different types based on behavior
    
    Args:
        trader_data: DataFrame with trader metrics
        n_clusters: Number of trader types to identify
        
    Returns:
        Dictionary with classification results
    """
    if len(trader_data) < n_clusters:
        print(f"Warning: Too few traders ({len(trader_data)}) for {n_clusters} clusters")
        n_clusters = min(n_clusters, max(2, len(trader_data) // 2))
        
    # Select features for clustering
    features = ['maker_ratio', 'total_trades', 'avg_trade_size', 'buy_sell_ratio']
    
    # Ensure all features exist
    for feature in features:
        if feature not in trader_data.columns:
            print(f"Warning: Missing feature {feature}, using default value")
            trader_data[feature] = 0.0
    
    # Handle infinite values
    X = trader_data[features].replace([np.inf, -np.inf], 10)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    trader_data['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster characteristics
    cluster_profiles = trader_data.groupby('cluster')[features + ['pnl']].mean()
    cluster_profiles['count'] = trader_data.groupby('cluster').size()
    cluster_profiles['percent'] = 100 * cluster_profiles['count'] / len(trader_data)
    
    # Assign readable labels to clusters
    trader_types = []
    for cluster_id, profile in cluster_profiles.iterrows():
        if profile['maker_ratio'] > 0.7:
            type_name = "Market Maker"
        elif profile['buy_sell_ratio'] > 2:
            type_name = "Momentum Buyer"
        elif profile['buy_sell_ratio'] < 0.5:
            type_name = "Momentum Seller"
        elif profile['total_trades'] > trader_data['total_trades'].median() * 2:
            type_name = "Active Trader"
        elif profile['avg_trade_size'] > trader_data['avg_trade_size'].median() * 2:
            type_name = "Whale"
        else:
            type_name = "Retail Trader"
            
        trader_types.append({
            'cluster': cluster_id,
            'type': type_name,
            'count': profile['count'],
            'percent': profile['percent']
        })
    
    # Create type summary
    type_summary = pd.DataFrame(trader_types)
    
    # Assign types back to traders
    cluster_to_type = {row['cluster']: row['type'] for row in trader_types}
    trader_data['type'] = trader_data['cluster'].map(cluster_to_type)
    
    return {
        'trader_data': trader_data,
        'cluster_profiles': cluster_profiles,
        'type_summary': type_summary
    }

if __name__ == "__main__":
    # Run analysis when script is executed directly
    analyze_traders(n_markets=3, save_path='results/trader_analysis')