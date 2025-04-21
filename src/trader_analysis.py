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

def load_data(data_path='data/cleaned_election_data.csv'):
    """
    Load the main election dataset with trader metrics
    
    Parameters:
    -----------
    data_path : str
        Path to the main dataset CSV
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with trader metrics
    """
    try:
        # Load with optimized settings
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_gini(values):
    """
    Calculate Gini coefficient (0=equal, 1=unequal)
    
    Parameters:
    -----------
    values : array-like
        Values to calculate Gini coefficient for
        
    Returns:
    --------
    float
        Gini coefficient
    """
    # Handle edge cases
    if len(values) <= 1 or values.sum() == 0:
        return 0
        
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Calculate using cumulative distribution
    cum_values = np.cumsum(sorted_values)
    return 1 - 2 * np.sum(cum_values / cum_values[-1]) / n + 1 / n

def plot_lorenz_curve(values, title, ax=None):
    """
    Plot Lorenz curve for inequality visualization
    
    Parameters:
    -----------
    values : array-like
        Values to plot Lorenz curve for
    title : str
        Title for the plot
    ax : matplotlib.axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes
        Axes with Lorenz curve
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle edge cases
    if len(values) <= 1 or values.sum() == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha='center')
        return ax
        
    sorted_values = np.sort(values)
    cumsum = np.cumsum(sorted_values)
    
    # Calculate normalized cumulative distribution
    y_lorenz = cumsum / cumsum[-1]
    x_lorenz = np.arange(1, len(values) + 1) / len(values)
    
    ax.plot(x_lorenz, y_lorenz, label='Lorenz curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect equality')
    ax.fill_between(x_lorenz, x_lorenz, y_lorenz, alpha=0.2)
    ax.set_title(f'Lorenz Curve - {title}')
    ax.set_xlabel('Cumulative % of traders')
    ax.set_ylabel(f'Cumulative % of {title}')
    ax.legend()
    
    return ax

def classify_traders(market_data, trader_features, n_clusters=5):
    """
    Classify traders into different types based on behavior
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        DataFrame with market data
    trader_features : list
        List of features to use for clustering
    n_clusters : int
        Number of clusters to create
        
    Returns:
    --------
    dict
        Dictionary with classification results
    """
    # Extract trader metrics
    trader_metrics = market_data[trader_features].copy()
    
    # Handle missing values
    trader_metrics.fillna(trader_metrics.mean(), inplace=True)
    
    # Replace infinite values
    trader_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
    trader_metrics.fillna(trader_metrics.mean(), inplace=True)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trader_metrics)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add clusters to data
    market_data_with_clusters = market_data.copy()
    market_data_with_clusters['cluster'] = clusters
    
    # Analyze cluster characteristics
    cluster_profiles = market_data_with_clusters.groupby('cluster')[trader_features].mean()
    cluster_profiles['count'] = market_data_with_clusters.groupby('cluster').size()
    cluster_profiles['percent'] = 100 * cluster_profiles['count'] / len(market_data_with_clusters)
    
    # Name clusters based on characteristics
    trader_types = []
    
    for cluster_id, profile in cluster_profiles.iterrows():
        # Extract key metrics
        maker_ratio = profile.get('buy_sell_ratio', 1.0)
        trade_count = profile.get('unique_traders_count', 0)
        avg_size = profile.get('trader_to_trade_ratio', 1.0)
        concentration = profile.get('trader_concentration', 0.0)
        
        # Assign type based on characteristics
        if concentration > 0.7:
            type_name = "Whale Traders"
        elif maker_ratio > 1.5:
            type_name = "Momentum Buyers"
        elif maker_ratio < 0.7:
            type_name = "Contrarian Sellers"
        elif trade_count > market_data['unique_traders_count'].median() * 1.5:
            type_name = "Active Traders"
        elif avg_size > market_data['trader_to_trade_ratio'].median() * 1.5:
            type_name = "High Frequency Traders"
        else:
            type_name = "Retail Traders"
        
        trader_types.append({
            'cluster': cluster_id,
            'type': type_name,
            'count': profile['count'],
            'percent': profile['percent']
        })
    
    # Create type summary
    type_summary = pd.DataFrame(trader_types)
    
    # Map cluster to type
    cluster_to_type = {row['cluster']: row['type'] for row in trader_types}
    market_data_with_clusters['trader_type'] = market_data_with_clusters['cluster'].map(cluster_to_type)
    
    return {
        'data_with_clusters': market_data_with_clusters,
        'cluster_profiles': cluster_profiles,
        'type_summary': type_summary
    }

def analyze_trader_concentration(market_data, min_markets=5, save_path='results/trader_analysis'):
    """
    Analyze trader concentration across markets
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        DataFrame with market data
    min_markets : int
        Minimum number of markets in a category to include in analysis
    save_path : str
        Path to save results
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Extract relevant columns
    if 'trader_concentration' not in market_data.columns:
        print("Warning: trader_concentration not found in data")
        # Try to calculate it if needed metrics are available
        if 'unique_traders_count' in market_data.columns and 'volumeNum' in market_data.columns:
            print("Calculating proxy for trader concentration")
            # This is just a proxy - real calculation would need trader-level data
            market_data['trader_concentration'] = 1 / (market_data['unique_traders_count'] / 
                                                     np.sqrt(market_data['volumeNum']))
    
    # Calculate Gini coefficients across different metrics
    concentration_metrics = {
        'Trader Count': calculate_gini(market_data['unique_traders_count'].dropna()),
        'Trading Volume': calculate_gini(market_data['volumeNum'].dropna() if 'volumeNum' in market_data.columns else []),
        'Trader Concentration': calculate_gini(market_data['trader_concentration'].dropna())
    }
    
    # Analyze concentration by election type
    type_concentration = {}
    if 'event_electionType' in market_data.columns:
        for election_type, group in market_data.groupby('event_electionType'):
            if len(group) >= min_markets:
                type_concentration[election_type] = {
                    'n_markets': len(group),
                    'trader_gini': calculate_gini(group['unique_traders_count'].dropna()),
                    'volume_gini': calculate_gini(group['volumeNum'].dropna() if 'volumeNum' in market_data.columns else []),
                    'concentration_gini': calculate_gini(group['trader_concentration'].dropna()),
                    'avg_traders': group['unique_traders_count'].mean(),
                    'avg_concentration': group['trader_concentration'].mean()
                }
    
    # Analyze concentration by country
    country_concentration = {}
    if 'event_country' in market_data.columns:
        for country, group in market_data.groupby('event_country'):
            if len(group) >= min_markets:
                country_concentration[country] = {
                    'n_markets': len(group),
                    'trader_gini': calculate_gini(group['unique_traders_count'].dropna()),
                    'volume_gini': calculate_gini(group['volumeNum'].dropna() if 'volumeNum' in market_data.columns else []),
                    'concentration_gini': calculate_gini(group['trader_concentration'].dropna()),
                    'avg_traders': group['unique_traders_count'].mean(),
                    'avg_concentration': group['trader_concentration'].mean()
                }
    
    # Create visualizations
    
    # 1. Overall concentration metrics
    plt.figure(figsize=(10, 6))
    metrics = list(concentration_metrics.keys())
    values = list(concentration_metrics.values())
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.title('Market Concentration Metrics (Gini Coefficients)')
    plt.ylabel('Gini Coefficient (0=Equal, 1=Unequal)')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(save_path, 'concentration_metrics.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Lorenz curves
    plt.figure(figsize=(15, 5))
    
    # Trader count
    plt.subplot(1, 3, 1)
    plot_lorenz_curve(market_data['unique_traders_count'].dropna(), 'Trader Count')
    
    # Volume
    plt.subplot(1, 3, 2)
    if 'volumeNum' in market_data.columns:
        plot_lorenz_curve(market_data['volumeNum'].dropna(), 'Trading Volume')
    else:
        plt.text(0.5, 0.5, "Volume data not available", ha='center')
    
    # Concentration
    plt.subplot(1, 3, 3)
    plot_lorenz_curve(market_data['trader_concentration'].dropna(), 'Trader Concentration')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lorenz_curves.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Concentration by election type
    if type_concentration:
        # Convert to DataFrame for easier plotting
        type_df = pd.DataFrame.from_dict(type_concentration, orient='index')
        
        plt.figure(figsize=(12, 6))
        type_df.sort_values('concentration_gini').plot(
            y='concentration_gini', kind='barh', 
            title='Trader Concentration by Election Type (Gini)', 
            xlabel='Gini Coefficient', figsize=(12, 6))
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'concentration_by_type.png'), bbox_inches='tight')
        plt.close()
    
    # 4. Concentration by country
    if country_concentration:
        # Convert to DataFrame for easier plotting
        country_df = pd.DataFrame.from_dict(country_concentration, orient='index')
        
        # Plot top 10 countries by market count
        top_countries = country_df.sort_values('n_markets', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        top_countries.sort_values('concentration_gini').plot(
            y='concentration_gini', kind='barh', 
            title='Trader Concentration by Country (Gini)', 
            xlabel='Gini Coefficient', figsize=(12, 6))
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'concentration_by_country.png'), bbox_inches='tight')
        plt.close()
    
    return {
        'overall_metrics': concentration_metrics,
        'by_election_type': type_concentration,
        'by_country': country_concentration
    }

def identify_whales(market_data, trade_data, threshold=0.05, method='volume'):
    """
    Identify whale traders based on specified criteria
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        DataFrame with market data
    trade_data : pd.DataFrame
        DataFrame with trade-level data
    threshold : float
        Threshold for defining whales (e.g., top 5%)
    method : str
        Method for identifying whales ('volume', 'frequency', or 'position_size')
        
    Returns:
    --------
    dict
        Dictionary with whale analysis results
    """
    # Group trades by trader
    if 'trader_id' not in trade_data.columns:
        print("Error: trader_id column not found in trade data")
        return None
    
    # Aggregate by trader based on method
    if method == 'volume':
        # Sum trade volume by trader
        trader_metrics = trade_data.groupby('trader_id')['trade_amount'].sum().reset_index()
        metric_name = 'total_volume'
    elif method == 'frequency':
        # Count trades by trader
        trader_metrics = trade_data.groupby('trader_id').size().reset_index(name='trade_count')
        metric_name = 'trade_count'
    elif method == 'position_size':
        # Calculate average position size by trader
        trader_metrics = trade_data.groupby('trader_id')['trade_amount'].mean().reset_index()
        metric_name = 'avg_position_size'
    else:
        print(f"Unknown method: {method}")
        return None
    
    # Sort traders by the metric
    trader_metrics = trader_metrics.sort_values(metric_name, ascending=False)
    
    # Define whales as top traders by threshold
    whale_count = max(1, int(len(trader_metrics) * threshold))
    whales = trader_metrics.head(whale_count)
    
    # Calculate whale concentration
    whale_concentration = whales[metric_name].sum() / trader_metrics[metric_name].sum()
    
    # Calculate average metric for whales vs non-whales
    whale_avg = whales[metric_name].mean()
    non_whale_avg = trader_metrics.iloc[whale_count:][metric_name].mean() if len(trader_metrics) > whale_count else 0
    
    return {
        'whales': whales,
        'whale_ids': whales['trader_id'].tolist(),
        'whale_count': whale_count,
        'total_trader_count': len(trader_metrics),
        'whale_concentration': whale_concentration,
        'whale_avg_metric': whale_avg,
        'non_whale_avg_metric': non_whale_avg,
        'whale_to_non_whale_ratio': whale_avg / non_whale_avg if non_whale_avg > 0 else float('inf')
    }

def analyze_whale_impact(market_data, trade_data, whale_ids, min_trades=10):
    """
    Analyze the impact of whale trades on market prices
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        DataFrame with market data
    trade_data : pd.DataFrame
        DataFrame with trade-level data including timestamps and prices
    whale_ids : list
        List of whale trader IDs
    min_trades : int
        Minimum number of trades required for analysis
        
    Returns:
    --------
    dict
        Dictionary with whale impact analysis results
    """
    # Filter for markets with sufficient trades
    market_trade_counts = trade_data.groupby('market_id').size()
    markets_with_sufficient_trades = market_trade_counts[market_trade_counts >= min_trades].index
    
    # If no markets have sufficient trades, return early
    if len(markets_with_sufficient_trades) == 0:
        print(f"No markets with at least {min_trades} trades found")
        return None
    
    results = {
        'markets_analyzed': len(markets_with_sufficient_trades),
        'whale_price_impact': [],
        'non_whale_price_impact': [],
        'whale_followed_ratio': []
    }
    
    # Analyze each market with sufficient trades
    for market_id in markets_with_sufficient_trades:
        market_trades = trade_data[trade_data['market_id'] == market_id].sort_values('timestamp')
        
        # Separate whale and non-whale trades
        whale_trades = market_trades[market_trades['trader_id'].isin(whale_ids)]
        non_whale_trades = market_trades[~market_trades['trader_id'].isin(whale_ids)]
        
        if len(whale_trades) == 0 or len(non_whale_trades) == 0:
            continue
        
        # Analyze price impact (simplified)
        whale_price_changes = []
        non_whale_price_changes = []
        whale_followed_count = 0
        whale_total_count = 0
        
        # Calculate average price change after whale trades
        for i, trade in whale_trades.iterrows():
            # Get next trade price
            next_trades = market_trades[market_trades['timestamp'] > trade['timestamp']].head(5)
            if len(next_trades) > 0:
                price_change = next_trades.iloc[-1]['price'] - trade['price']
                whale_price_changes.append(price_change)
                
                # Check if non-whales follow whale direction
                whale_total_count += 1
                next_non_whale = non_whale_trades[non_whale_trades['timestamp'] > trade['timestamp']].head(3)
                if len(next_non_whale) > 0:
                    # Simplified: check if trade direction (buy/sell) matches
                    if 'trade_direction' in trade and 'trade_direction' in next_non_whale.iloc[0]:
                        if trade['trade_direction'] == next_non_whale.iloc[0]['trade_direction']:
                            whale_followed_count += 1
        
        # Calculate average price change after non-whale trades
        for i, trade in non_whale_trades.iterrows():
            next_trades = market_trades[market_trades['timestamp'] > trade['timestamp']].head(5)
            if len(next_trades) > 0:
                price_change = next_trades.iloc[-1]['price'] - trade['price']
                non_whale_price_changes.append(price_change)
        
        # Calculate average price impacts
        if whale_price_changes:
            results['whale_price_impact'].append(np.mean(whale_price_changes))
        if non_whale_price_changes:
            results['non_whale_price_impact'].append(np.mean(non_whale_price_changes))
        
        # Calculate following ratio
        if whale_total_count > 0:
            results['whale_followed_ratio'].append(whale_followed_count / whale_total_count)
    
    # Calculate summary statistics
    if results['whale_price_impact']:
        results['avg_whale_price_impact'] = np.mean(results['whale_price_impact'])
    if results['non_whale_price_impact']:
        results['avg_non_whale_price_impact'] = np.mean(results['non_whale_price_impact'])
    if results['whale_followed_ratio']:
        results['avg_whale_followed_ratio'] = np.mean(results['whale_followed_ratio'])
    
    return results

def run_granger_causality_test(market_id, trade_data, whale_ids, max_lag=5):
    """
    Perform Granger causality test to determine if whale trades cause price movements
    
    Parameters:
    -----------
    market_id : str or int
        ID of the market to analyze
    trade_data : pd.DataFrame
        DataFrame with trade-level data
    whale_ids : list
        List of whale trader IDs
    max_lag : int
        Maximum number of lags for Granger test
        
    Returns:
    --------
    dict
        Dictionary with Granger causality test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Filter trades for this market
    market_trades = trade_data[trade_data['market_id'] == market_id].sort_values('timestamp')
    
    if len(market_trades) < max_lag + 10:
        print(f"Insufficient trades for market {market_id}")
        return None
    
    # Create time series by resampling to regular intervals
    market_trades['timestamp'] = pd.to_datetime(market_trades['timestamp'])
    market_trades = market_trades.set_index('timestamp')
    
    # Resample to 5-minute intervals
    prices = market_trades['price'].resample('5Min').last().dropna()
    
    # Create indicator for whale activity
    whale_trades = market_trades[market_trades['trader_id'].isin(whale_ids)]
    whale_activity = whale_trades.groupby(pd.Grouper(freq='5Min')).size()
    
    # Align the series
    aligned_data = pd.concat([prices, whale_activity], axis=1).fillna(0)
    aligned_data.columns = ['price', 'whale_activity']
    
    if len(aligned_data) < max_lag + 10:
        print(f"Insufficient data points after resampling for market {market_id}")
        return None
    
    # Run Granger causality test
    try:
        # Test if whale activity Granger-causes prices
        gc_result = grangercausalitytests(aligned_data[['price', 'whale_activity']], maxlag=max_lag, verbose=False)
        
        # Extract p-values for each lag
        p_values = {lag: result[0]['ssr_chi2test'][1] for lag, result in gc_result.items()}
        
        # Check for significance
        significant_lags = [lag for lag, p_value in p_values.items() if p_value < 0.05]
        
        return {
            'market_id': market_id,
            'p_values': p_values,
            'significant_lags': significant_lags,
            'is_significant': len(significant_lags) > 0,
            'min_p_value': min(p_values.values()) if p_values else None,
            'best_lag': min(p_values, key=p_values.get) if p_values else None
        }
    except Exception as e:
        print(f"Error running Granger causality test for market {market_id}: {e}")
        return None

def classify_traders_by_behavior(trade_data, n_clusters=5, random_state=42):
    """
    Classify traders into different types based on their behavior patterns
    
    Parameters:
    -----------
    trade_data : pd.DataFrame
        DataFrame with trade-level data
    n_clusters : int
        Number of clusters to create
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with classification results
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Ensure we have trader_id column
    if 'trader_id' not in trade_data.columns:
        print("Error: trader_id column not found in trade data")
        return None
    
    # Calculate trader features
    traders = []
    unique_traders = trade_data['trader_id'].unique()
    
    for trader_id in unique_traders:
        trader_trades = trade_data[trade_data['trader_id'] == trader_id]
        
        # Skip traders with too few trades
        if len(trader_trades) < 3:
            continue
        
        # Calculate basic metrics
        trade_count = len(trader_trades)
        
        # Calculate volume metrics if available
        avg_trade_size = trader_trades['trade_amount'].mean() if 'trade_amount' in trader_trades.columns else np.nan
        total_volume = trader_trades['trade_amount'].sum() if 'trade_amount' in trader_trades.columns else np.nan
        
        # Calculate timing metrics
        if 'timestamp' in trader_trades.columns:
            trader_trades = trader_trades.sort_values('timestamp')
            time_between_trades = np.diff(trader_trades['timestamp'].astype(np.int64)).mean() / 1e9 if len(trader_trades) > 1 else np.nan
        else:
            time_between_trades = np.nan
        
        # Calculate directional bias if available
        if 'trade_direction' in trader_trades.columns:
            # Assuming 1 = buy, -1 = sell
            direction_values = trader_trades['trade_direction'].map({'buy': 1, 'sell': -1, 1: 1, -1: -1}).dropna()
            directional_bias = direction_values.mean() if len(direction_values) > 0 else 0
        else:
            directional_bias = 0
        
        # Calculate profit metrics if available
        profit = trader_trades['profit'].sum() if 'profit' in trader_trades.columns else np.nan
        win_rate = (trader_trades['profit'] > 0).mean() if 'profit' in trader_trades.columns else np.nan
        
        # Calculate market diversity
        market_count = trader_trades['market_id'].nunique() if 'market_id' in trader_trades.columns else 1
        market_concentration = (trader_trades.groupby('market_id').size() / trade_count).max() if 'market_id' in trader_trades.columns else 1
        
        # Store trader features
        traders.append({
            'trader_id': trader_id,
            'trade_count': trade_count,
            'avg_trade_size': avg_trade_size,
            'total_volume': total_volume,
            'time_between_trades': time_between_trades,
            'directional_bias': directional_bias,
            'profit': profit,
            'win_rate': win_rate,
            'market_count': market_count,
            'market_concentration': market_concentration
        })
    
    # Convert to DataFrame
    trader_df = pd.DataFrame(traders)
    
    # Handle missing values
    trader_df = trader_df.fillna(trader_df.mean())
    
    # Select features for clustering
    feature_cols = [col for col in ['trade_count', 'avg_trade_size', 'time_between_trades', 
                                    'directional_bias', 'market_count', 'market_concentration'] 
                   if col in trader_df.columns]
    
    if len(feature_cols) < 2:
        print("Error: Insufficient features for clustering")
        return None
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trader_df[feature_cols])
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    trader_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Analyze clusters
    cluster_profiles = trader_df.groupby('cluster')[feature_cols].mean()
    
    # Determine cluster sizes
    cluster_sizes = trader_df['cluster'].value_counts().sort_index()
    cluster_percentages = 100 * cluster_sizes / cluster_sizes.sum()
    
    # Add size information to profiles
    cluster_profiles['count'] = cluster_sizes.values
    cluster_profiles['percentage'] = cluster_percentages.values
    
    # Name clusters based on features
    cluster_names = {}
    for cluster_id, profile in cluster_profiles.iterrows():
        # Start with generic name
        name = f"Cluster {cluster_id}"
        
        # Determine distinguishing features
        if 'trade_count' in profile and profile['trade_count'] > cluster_profiles['trade_count'].median() * 2:
            name = "High Frequency Traders"
        elif 'avg_trade_size' in profile and profile['avg_trade_size'] > cluster_profiles['avg_trade_size'].median() * 2:
            name = "Whale Traders"
        elif 'directional_bias' in profile:
            if profile['directional_bias'] > 0.5:
                name = "Bullish Traders"
            elif profile['directional_bias'] < -0.5:
                name = "Bearish Traders"
            elif abs(profile['directional_bias']) < 0.2:
                name = "Market Makers"
        elif 'market_concentration' in profile and profile['market_concentration'] > 0.8:
            name = "Market Specialists"
        elif 'market_count' in profile and profile['market_count'] > cluster_profiles['market_count'].median() * 2:
            name = "Diversified Traders"
        
        cluster_names[cluster_id] = name
    
    # Assign names to traders
    trader_df['trader_type'] = trader_df['cluster'].map(cluster_names)
    
    return {
        'trader_features': trader_df,
        'cluster_profiles': cluster_profiles,
        'cluster_names': cluster_names,
        'feature_importance': {feature: abs(kmeans.cluster_centers_).mean(axis=0)[i] 
                              for i, feature in enumerate(feature_cols)}
    }

def analyze_traders(n_markets=None, data_path='data/cleaned_election_data.csv', save_path='results/trader_analysis'):
    """
    Run comprehensive trader analysis
    
    Parameters:
    -----------
    n_markets : int, optional
        Number of markets to analyze (None = all)
    data_path : str
        Path to the main dataset
    save_path : str
        Path to save results
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving results to {save_path}")
    
    # Load data
    market_data = load_data(data_path)
    if market_data is None:
        print("Failed to load data")
        return None
    
    # Filter to top N markets by volume if specified
    if n_markets is not None:
        if 'volumeNum' in market_data.columns:
            market_data = market_data.sort_values('volumeNum', ascending=False).head(n_markets)
            print(f"Analyzing top {n_markets} markets by volume")
        else:
            market_data = market_data.head(n_markets)
            print(f"Analyzing first {n_markets} markets (volume data not available)")
    
    # Check for required trader metrics
    required_metrics = ['unique_traders_count', 'trader_to_trade_ratio']
    missing_metrics = [metric for metric in required_metrics if metric not in market_data.columns]
    
    if missing_metrics:
        print(f"Warning: Missing required trader metrics: {missing_metrics}")
        return None
    
    print(f"Analyzing {len(market_data)} markets")
    
    # Define trader features for classification
    trader_features = [
        'unique_traders_count',
        'trader_to_trade_ratio',
        'two_way_traders_ratio',
        'trader_concentration',
        'buy_sell_ratio',
        'trading_frequency'
    ]
    
    # Filter out any missing features
    available_features = [feature for feature in trader_features if feature in market_data.columns]
    print(f"Using available trader features: {available_features}")
    
    # Analyze trader concentration
    concentration_results = analyze_trader_concentration(market_data, save_path=save_path)
    
    # Classify trader types (if enough features are available)
    trader_types = None
    if len(available_features) >= 3:
        trader_types = classify_traders(market_data, available_features)
        
        # Save trader type summary
        trader_types['type_summary'].to_csv(os.path.join(save_path, 'trader_types.csv'), index=False)
        
        # Create trader type visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x='type', y='percent', data=trader_types['type_summary'])
        plt.title('Trader Type Distribution')
        plt.xlabel('Trader Type')
        plt.ylabel('Percentage of Markets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'trader_types.png'), bbox_inches='tight')
        plt.close()
        
        # Create profile visualization
        plt.figure(figsize=(12, 8))
        profile_df = trader_types['cluster_profiles'].copy()
        
        # Normalize features for comparison
        for col in available_features:
            if col in profile_df.columns:
                profile_df[col] = profile_df[col] / profile_df[col].max()
        
        # Plot heatmap of normalized profiles
        sns.heatmap(profile_df[available_features], annot=True, cmap='viridis', fmt='.2f')
        plt.title('Trader Type Profiles (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'trader_profiles.png'), bbox_inches='tight')
        plt.close()
    
    # Create market summary
    market_summary = market_data[['id', 'question']].copy() if 'question' in market_data.columns else market_data[['id']].copy()
    
    # Add key metrics
    for metric in ['unique_traders_count', 'trader_concentration', 'trader_to_trade_ratio', 'buy_sell_ratio']:
        if metric in market_data.columns:
            market_summary[metric] = market_data[metric]
    
    if 'volumeNum' in market_data.columns:
        market_summary['volume'] = market_data['volumeNum']
    
    if 'brier_score' in market_data.columns:
        market_summary['prediction_accuracy'] = 1 - market_data['brier_score']
    
    # Save market summary
    market_summary.to_csv(os.path.join(save_path, 'market_summary.csv'), index=False)
    
    # Return combined results
    return {
        'market_summary': market_summary,
        'concentration': concentration_results,
        'trader_types': trader_types
    }

if __name__ == "__main__":
    # Run analysis when script is executed directly
    analyze_traders(n_markets=100, save_path='results/trader_analysis')