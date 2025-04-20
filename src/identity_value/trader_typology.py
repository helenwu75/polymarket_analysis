#!/usr/bin/env python3
'''
Trader Typology

This module will contain code for analyzing Polymarket data focusing on trader typology.
'''

def classify_trader_types(market_df, trades_dir='data/trades', min_trades=5):
    """
    Classify traders into types based on their behavior patterns
    """
    all_trader_features = []
    market_ids = market_df['market_id'].unique()
    
    for market_id in market_ids:
        trades_df = load_trade_data(market_id, trades_dir)
        if trades_df is None or len(trades_df) < min_trades:
            continue
        
        # Calculate market start and end dates
        market_start = trades_df['timestamp'].min()
        market_end = trades_df['timestamp'].max()
        market_duration = (market_end - market_start).total_seconds() / (24 * 3600)  # in days
        
        # Identify all traders
        makers = set(trades_df['maker'].unique())
        takers = set(trades_df['taker'].unique())
        all_traders = makers.union(takers)
        
        for trader in all_traders:
            # Get trader's maker and taker trades
            maker_trades = trades_df[trades_df['maker'] == trader]
            taker_trades = trades_df[trades_df['taker'] == trader]
            
            all_trades = pd.concat([
                maker_trades[['timestamp', 'maker', 'makerAmountFilled']].rename(
                    columns={'maker': 'trader', 'makerAmountFilled': 'amount'}),
                taker_trades[['timestamp', 'taker', 'takerAmountFilled']].rename(
                    columns={'taker': 'trader', 'takerAmountFilled': 'amount'})
            ])
            
            if len(all_trades) < min_trades:
                continue
                
            # Calculate trader features
            first_trade = all_trades['timestamp'].min()
            last_trade = all_trades['timestamp'].max()
            
            # Timing features
            entry_timing = (first_trade - market_start).total_seconds() / (market_duration * 24 * 3600)
            duration_ratio = (last_trade - first_trade).total_seconds() / (market_duration * 24 * 3600)
            
            # Trade size features
            avg_trade_size = all_trades['amount'].mean()
            max_trade_size = all_trades['amount'].max()
            trade_size_cv = all_trades['amount'].std() / all_trades['amount'].mean() if all_trades['amount'].mean() > 0 else 0
            
            # Trading frequency
            total_trades = len(all_trades)
            trades_per_day = total_trades / market_duration if market_duration > 0 else 0
            
            # Maker/Taker ratio
            maker_ratio = len(maker_trades) / total_trades if total_trades > 0 else 0
            
            trader_features = {
                'market_id': market_id,
                'trader_id': trader,
                'total_trades': total_trades,
                'entry_timing': entry_timing,
                'duration_ratio': duration_ratio,
                'avg_trade_size': avg_trade_size,
                'max_trade_size': max_trade_size,
                'trade_size_cv': trade_size_cv,
                'trades_per_day': trades_per_day,
                'maker_ratio': maker_ratio
            }
            
            all_trader_features.append(trader_features)
    
    trader_features_df = pd.DataFrame(all_trader_features)
    
    # Cluster traders using K-means
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    if len(trader_features_df) == 0:
        print("No trader features collected. Ensure min_trades is not too high.")
        return None
    
    # Select features for clustering
    cluster_features = [
        'entry_timing', 'duration_ratio', 'avg_trade_size', 
        'trade_size_cv', 'trades_per_day', 'maker_ratio'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trader_features_df[cluster_features])
    
    # Determine optimal number of clusters using silhouette score
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    k_range = range(2, min(11, len(trader_features_df)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        
        if len(set(labels)) < 2:  # Skip if only one cluster is formed
            silhouette_scores.append(0)
            continue
            
        score = silhouette_score(scaled_features, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Cluster with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    trader_features_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Characterize clusters
    cluster_profiles = trader_features_df.groupby('cluster')[cluster_features].mean()
    
    # Assign trader types based on cluster characteristics
    trader_types = []
    for cluster_id, profile in cluster_profiles.iterrows():
        if profile['trades_per_day'] > 10 and profile['maker_ratio'] > 0.7:
            trader_type = "Market Maker"
        elif profile['entry_timing'] < 0.2 and profile['duration_ratio'] > 0.7:
            trader_type = "Long-term Holder"
        elif profile['trade_size_cv'] > 1.5 and profile['avg_trade_size'] > trader_features_df['avg_trade_size'].mean() * 2:
            trader_type = "Whale"
        elif profile['duration_ratio'] < 0.1 and profile['trades_per_day'] > 5:
            trader_type = "Day Trader"
        elif profile['entry_timing'] > 0.8:
            trader_type = "Late Entrant"
        else:
            trader_type = f"Cluster {cluster_id}"
        
        trader_types.append((cluster_id, trader_type))
    
    trader_type_map = dict(trader_types)
    trader_features_df['trader_type'] = trader_features_df['cluster'].map(trader_type_map)
    
    return trader_features_df, cluster_profiles, trader_type_map

def plot_trader_types(trader_features_df, cluster_profiles, trader_type_map, save_path=None):
    """
    Plot visualization of trader types and their characteristics
    """
    # PCA for 2D visualization
    from sklearn.decomposition import PCA
    
    cluster_features = [
        'entry_timing', 'duration_ratio', 'avg_trade_size', 
        'trade_size_cv', 'trades_per_day', 'maker_ratio'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trader_features_df[cluster_features])
    
    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    