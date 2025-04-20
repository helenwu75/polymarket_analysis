#!/usr/bin/env python3
"""
Trader Typology Analysis

This script implements the identity value analysis focusing on trader typology.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our modified data loader functions
from utils.data_loader import (
    load_main_dataset,
    load_market_question_mapping,
    load_trade_data,
    get_sample_market_ids
)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def analyze_trader_typology(sample_size=5, min_trades=10, verbose=True):
    """
    Analyze trader types based on behavior patterns
    
    Args:
        sample_size: Number of markets to analyze
        min_trades: Minimum number of trades required for a trader to be included
        verbose: Whether to print detailed results
    """
    print("\n=== Trader Typology Analysis ===")
    
    # Get sample market IDs
    market_ids = get_sample_market_ids(sample_size)
    market_questions = load_market_question_mapping()
    
    # Results directory
    results_dir = 'results/identity_value'
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect trader features across all markets
    all_trader_features = []
    analyzed_markets = []
    
    for market_id in market_ids:
        market_name = market_questions.get(market_id, f"Market {market_id}")
        if verbose:
            print(f"\nAnalyzing market: {market_name}")
        
        # Load trade data
        trades_df = load_trade_data(market_id)
        if trades_df is None or len(trades_df) < 100:
            if verbose:
                print(f"Insufficient data for market {market_id}, skipping...")
            continue
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Calculate market start and end dates
        market_start = trades_df['timestamp'].min()
        market_end = trades_df['timestamp'].max()
        market_duration = (market_end - market_start).total_seconds() / (24 * 3600)  # in days
        
        analyzed_markets.append({
            'market_id': market_id,
            'market_name': market_name,
            'n_trades': len(trades_df),
            'start_date': market_start,
            'end_date': market_end,
            'duration_days': market_duration
        })
        
        if verbose:
            print(f"  Trades: {len(trades_df)}")
            print(f"  Period: {market_start.date()} to {market_end.date()} ({market_duration:.1f} days)")
        
        # Identify all traders
        makers = set(trades_df['maker_id'].unique())
        takers = set(trades_df['taker_id'].unique())
        all_traders = makers.union(takers)
        
        # Ensure numeric columns
        trades_df['volume'] = pd.to_numeric(trades_df['size'])
        trades_df['price_num'] = pd.to_numeric(trades_df['price'])
        
        # Process each trader
        for trader in all_traders:
            # Get trader's maker and taker trades
            maker_trades = trades_df[trades_df['maker_id'] == trader]
            taker_trades = trades_df[trades_df['taker_id'] == trader]
            
            all_trades = pd.concat([
                maker_trades[['timestamp', 'volume', 'price_num', 'side']].assign(role='maker'),
                taker_trades[['timestamp', 'volume', 'price_num', 'side']].assign(role='taker')
            ])
            
            if len(all_trades) < min_trades:
                continue
                
            # Calculate trader features
            first_trade = all_trades['timestamp'].min()
            last_trade = all_trades['timestamp'].max()
            
            # Timing features
            entry_timing = (first_trade - market_start).total_seconds() / (market_duration * 24 * 3600) if market_duration > 0 else 0
            duration_ratio = (last_trade - first_trade).total_seconds() / (market_duration * 24 * 3600) if market_duration > 0 else 0
            
            # Trade size features
            avg_trade_size = all_trades['volume'].mean()
            max_trade_size = all_trades['volume'].max()
            trade_size_cv = all_trades['volume'].std() / all_trades['volume'].mean() if all_trades['volume'].mean() > 0 else 0
            
            # Trading frequency
            total_trades = len(all_trades)
            trades_per_day = total_trades / market_duration if market_duration > 0 else 0
            
            # Maker/Taker ratio
            maker_ratio = len(maker_trades) / total_trades if total_trades > 0 else 0
            
            # Buy/Sell ratio
            buy_trades = all_trades[all_trades['side'] == 'Buy']
            sell_trades = all_trades[all_trades['side'] == 'Sell']
            buy_sell_ratio = len(buy_trades) / len(sell_trades) if len(sell_trades) > 0 else float('inf')
            if buy_sell_ratio == float('inf'):
                buy_sell_ratio = 10.0  # Cap for analysis
            
            # Trading strategy features
            avg_price = all_trades['price_num'].mean()
            price_stdev = all_trades['price_num'].std()
            
            trader_features = {
                'market_id': market_id,
                'market_name': market_name,
                'trader_id': trader,
                'total_trades': total_trades,
                'entry_timing': entry_timing,
                'duration_ratio': duration_ratio,
                'avg_trade_size': avg_trade_size,
                'max_trade_size': max_trade_size,
                'trade_size_cv': trade_size_cv,
                'trades_per_day': trades_per_day,
                'maker_ratio': maker_ratio,
                'buy_sell_ratio': buy_sell_ratio,
                'avg_price': avg_price,
                'price_stdev': price_stdev
            }
            
            all_trader_features.append(trader_features)
    
    if not all_trader_features:
        print("No trader features collected. Try reducing min_trades or increasing sample_size.")
        return None
    
    # Create DataFrame of trader features
    trader_features_df = pd.DataFrame(all_trader_features)
    
    if verbose:
        print(f"\nCollected features for {len(trader_features_df)} traders across {len(analyzed_markets)} markets")
    
    # Select features for clustering
    cluster_features = [
        'entry_timing', 'duration_ratio', 'avg_trade_size', 
        'trade_size_cv', 'trades_per_day', 'maker_ratio',
        'buy_sell_ratio'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trader_features_df[cluster_features])
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    k_range = range(2, min(11, len(trader_features_df)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        if len(set(cluster_labels)) < 2:  # Skip if only one cluster is formed
            silhouette_scores.append(0)
            continue
            
        score = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores.append(score)
    
    if not silhouette_scores:
        print("Could not determine optimal number of clusters.")
        return None
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    if verbose:
        print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Cluster with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    trader_features_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Characterize clusters
    cluster_profiles = trader_features_df.groupby('cluster')[cluster_features].mean()
    cluster_profiles['n_traders'] = trader_features_df.groupby('cluster').size()
    
    # Add cluster sizes as percentage
    total_traders = len(trader_features_df)
    cluster_profiles['pct_traders'] = cluster_profiles['n_traders'] / total_traders * 100
    
    # Assign trader types based on cluster characteristics
    trader_types = []
    for cluster_id, profile in cluster_profiles.iterrows():
        if profile['trades_per_day'] > 5 and profile['maker_ratio'] > 0.7:
            trader_type = "Market Maker"
        elif profile['entry_timing'] < 0.2 and profile['duration_ratio'] > 0.7:
            trader_type = "Long-term Holder"
        elif profile['trade_size_cv'] > 1.5 and profile['avg_trade_size'] > trader_features_df['avg_trade_size'].mean() * 2:
            trader_type = "Whale"
        elif profile['duration_ratio'] < 0.2 and profile['trades_per_day'] > 3:
            trader_type = "Day Trader"
        elif profile['entry_timing'] > 0.8:
            trader_type = "Late Entrant"
        elif profile['buy_sell_ratio'] > 5:
            trader_type = "Momentum Buyer"
        elif profile['buy_sell_ratio'] < 0.2:
            trader_type = "Momentum Seller"
        else:
            trader_type = f"General Trader (Cluster {cluster_id})"
        
        trader_types.append((cluster_id, trader_type))
    
    trader_type_map = dict(trader_types)
    
    # Add trader type to profile
    cluster_profiles['trader_type'] = [trader_type_map[cluster_id] for cluster_id in cluster_profiles.index]
    
    # Add trader type to each trader
    trader_features_df['trader_type'] = trader_features_df['cluster'].map(trader_type_map)
    
    if verbose:
        print("\nCluster Profiles:")
        for cluster_id, profile in cluster_profiles.iterrows():
            print(f"\nCluster {cluster_id} - {profile['trader_type']} ({profile['n_traders']} traders, {profile['pct_traders']:.1f}%):")
            for feature in cluster_features:
                print(f"  {feature}: {profile[feature]:.4f}")
    
    # Plot 2D PCA visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = trader_features_df['cluster']
    pca_df['trader_type'] = trader_features_df['trader_type']
    
    # Create PCA plot
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster
    for cluster_id, trader_type in trader_type_map.items():
        cluster_data = pca_df[pca_df['cluster'] == cluster_id]
        plt.scatter(
            cluster_data['PC1'], 
            cluster_data['PC2'], 
            label=f"{trader_type} ({len(cluster_data)} traders)",
            alpha=0.7,
            s=50
        )
    
    plt.title('Trader Types - PCA Visualization', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Trader Type', fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'trader_typology_pca.png'), dpi=300, bbox_inches='tight')
    
    # Create radar chart for cluster profiles
    plt.figure(figsize=(15, 10))
    
    # Number of variables
    categories = cluster_features
    N = len(categories)
    
    # Create angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot for each cluster
    nrows = (optimal_k + 1) // 2  # Calculate rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    # Plot each cluster
    for i, (cluster_id, profile) in enumerate(cluster_profiles.iterrows()):
        if i < len(axes):
            ax = axes[i]
            
            # Scale the data for better visualization
            values = profile[categories].values.tolist()
            values += values[:1]  # Close the loop
            
            # Draw the radar chart
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            # Add feature labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            
            # Add title
            ax.set_title(f"{profile['trader_type']} (Cluster {cluster_id}: {profile['n_traders']} traders)", 
                        fontsize=12, pad=20)
    
    # Hide any unused subplots
    for i in range(len(cluster_profiles), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'trader_typology_radar.png'), dpi=300, bbox_inches='tight')
    
    # Save results to CSV
    trader_features_df.to_csv(os.path.join(results_dir, 'trader_features.csv'), index=False)
    cluster_profiles.to_csv(os.path.join(results_dir, 'cluster_profiles.csv'))
    
    print(f"\nResults saved to {results_dir}")
    
    return trader_features_df, cluster_profiles, trader_type_map

if __name__ == "__main__":
    analyze_trader_typology(sample_size=5, min_trades=10, verbose=True)