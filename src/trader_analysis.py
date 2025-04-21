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