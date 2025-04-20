import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from typing import Dict, List, Union, Optional, Tuple
import pyarrow.parquet as pq

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_main_dataset(filepath='data/cleaned_election_data.csv') -> pd.DataFrame:
    """
    Load the main dataset with market features
    """
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def load_market_question_mapping(filepath='data/trades/market_id_to_question.json') -> Dict:
    """
    Load mapping between market IDs and questions
    """
    with open(filepath, 'r') as f:
        mapping = json.load(f)
    return mapping

def load_market_tokens(filepath='data/trades/market_tokens.json') -> Dict:
    """
    Load mapping between market IDs and token IDs
    """
    with open(filepath, 'r') as f:
        mapping = json.load(f)
    return mapping

def find_token_id_file(token_id: str, trades_dir='data/trades/trades') -> Optional[str]:
    """
    Find a parquet file that matches a given token ID
    
    Args:
        token_id: The token ID to search for
        trades_dir: Directory containing trade data files
    
    Returns:
        Path to the matching file or None if no match found
    """
    # First try exact match
    for filename in os.listdir(trades_dir):
        if filename.startswith(token_id):
            return os.path.join(trades_dir, filename)
    
    # If no exact match, return None
    return None

def get_token_ids_for_market(market_id: Union[str, int, float], 
                             market_tokens_filepath='data/trades/market_tokens.json') -> List[str]:
    """
    Get token IDs associated with a market ID
    
    Args:
        market_id: The market ID to look up
        market_tokens_filepath: Path to the market tokens mapping file
    
    Returns:
        List of token IDs for the market
    """
    try:
        # Convert market_id to string if it's numeric
        if isinstance(market_id, (int, float)):
            market_id = str(int(market_id))
        
        # Load market tokens mapping
        with open(market_tokens_filepath, 'r') as f:
            market_tokens = json.load(f)
        
        # Check if market ID is in the mapping
        if market_id in market_tokens:
            return market_tokens[market_id]
        
        # If not, check if it's an Ethereum address with different capitalization
        if isinstance(market_id, str) and market_id.startswith('0x'):
            # Try case-insensitive match
            for key in market_tokens:
                if key.lower() == market_id.lower():
                    return market_tokens[key]
    except Exception as e:
        print(f"Error getting token IDs for market {market_id}: {e}")
    
    return []

def load_trade_data(market_id: Union[str, int, float], 
                    trades_dir='data/trades', 
                    return_all_tokens=True) -> Optional[pd.DataFrame]:
    """
    Load trade-level data for a specific market
    
    Args:
        market_id: The market ID to load data for
        trades_dir: Base directory for trade data
        return_all_tokens: If True, returns trades for all tokens in the market
    
    Returns:
        DataFrame with trade data or None if not found
    """
    # Get token IDs for this market
    token_ids = get_token_ids_for_market(market_id)
    
    if not token_ids:
        print(f"No token IDs found for market {market_id}")
        
        # If we don't have token IDs, try a random sample file for testing
        trades_subdir = os.path.join(trades_dir, "trades")
        if os.path.exists(trades_subdir):
            parquet_files = [f for f in os.listdir(trades_subdir) if f.endswith('.parquet')]
            if parquet_files:
                # Just return the first file for testing
                sample_file = os.path.join(trades_subdir, parquet_files[0])
                print(f"Returning sample file for testing: {os.path.basename(sample_file)}")
                df = pq.read_table(sample_file).to_pandas()
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                return df
        
        return None
    
    # List to store DataFrames for each token
    all_token_dfs = []
    
    # Load data for each token
    for token_id in token_ids:
        # Find the file that matches this token ID
        token_file = find_token_id_file(token_id, os.path.join(trades_dir, "trades"))
        
        if token_file and os.path.exists(token_file):
            try:
                # Load the parquet file
                df = pq.read_table(token_file).to_pandas()
                
                # Add token information
                df['market_id'] = market_id
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')

                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                all_token_dfs.append(df)
            except Exception as e:
                print(f"Error loading file for token {token_id}: {e}")
    
    if not all_token_dfs:
        print(f"No trade data found for market {market_id}")
        return None
    
    # Combine all token DataFrames
    if return_all_tokens:
        combined_df = pd.concat(all_token_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        return combined_df
    else:
        # Return the DataFrame for the first token (typically "Yes" token)
        return all_token_dfs[0]

def get_market_id_from_ethereum_address(ethereum_address: str, 
                                        mapping_file='data/trades/market_id_to_question.json') -> Optional[str]:
    """
    Get a market ID that corresponds to an Ethereum address
    
    Args:
        ethereum_address: Ethereum address to look up
        mapping_file: Path to the mapping file
    
    Returns:
        Market ID if found, None otherwise
    """
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        if ethereum_address in mapping:
            return ethereum_address
        
        # Try case-insensitive match
        ethereum_address_lower = ethereum_address.lower()
        for key in mapping:
            if isinstance(key, str) and key.lower() == ethereum_address_lower:
                return key
    except Exception as e:
        print(f"Error getting market ID from Ethereum address: {e}")
    
    return None

def get_sample_market_ids(n=5, mapping_file='data/trades/market_id_to_question.json') -> List[str]:
    """
    Get a sample of valid market IDs from the mapping file
    
    Args:
        n: Number of IDs to return
        mapping_file: Path to the mapping file
    
    Returns:
        List of market IDs
    """
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Filter out any non-string keys or NaN values
        valid_keys = [k for k in mapping.keys() if isinstance(k, str) and k != 'NaN']
        
        # Return up to n valid keys
        return valid_keys[:min(n, len(valid_keys))]
    except Exception as e:
        print(f"Error getting sample market IDs: {e}")
        return []

def summarize_dataset(df):
    """
    Print summary statistics and information about the dataset
    """
    print("Dataset summary:")
    print(f"Number of markets: {df.shape[0]}")
    
    if 'prediction_correct' in df.columns:
        print(f"Prediction accuracy: {df['prediction_correct'].mean()*100:.2f}%")
    
    if 'brier_score' in df.columns:
        print(f"Average Brier score: {df['brier_score'].mean():.4f}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\nColumns with missing values:")
    print(missing[missing > 0].sort_values(ascending=False))
    
    # Print summary of key features
    print("\nSummary of key features:")
    key_features = [col for col in ['brier_score', 'log_loss', 'price_range', 'price_volatility', 
                    'final_week_momentum', 'volumeNum', 'unique_traders_count', 
                    'trader_concentration', 'buy_sell_ratio'] if col in df.columns]
    
    if key_features:
        print(df[key_features].describe())
    
    # Distribution of markets by election type and region
    if 'event_electionType' in df.columns:
        print("\nDistribution by election type:")
        print(df['event_electionType'].value_counts())
    
    if 'event_country' in df.columns:
        print("\nDistribution by country:")
        print(df['event_country'].value_counts().head(10))