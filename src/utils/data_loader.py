import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Union, Optional, Tuple
import pyarrow.parquet as pq


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

def get_token_ids_for_market(market_id: Union[str, int, float], 
                             main_df=None) -> List[str]:
    """
    Get token IDs associated with a market ID
    
    Args:
        market_id: The market ID to look up
        main_df: DataFrame containing market data
    
    Returns:
        List of token IDs for the market
    """
    # If main_df not provided, load it
    if main_df is None:
        main_df = load_main_dataset()
    
    token_ids = []
    
    try:
        # Convert market_id to match DataFrame's ID column type
        market_id = main_df['id'].dtype.type(market_id)
        
        # Find the row for this market ID
        market_row = main_df[main_df['id'] == market_id]
        
        if not market_row.empty:
            # Extract clobTokenIds
            token_ids_str = market_row.iloc[0]['clobTokenIds']
            
            # Parse token IDs
            try:
                # Use json to parse the string representation of list
                token_ids = json.loads(token_ids_str)
            except json.JSONDecodeError:
                # Fallback to ast.literal_eval if json fails
                import ast
                token_ids = ast.literal_eval(token_ids_str)
            
            return token_ids
        else:
            # BACKUP METHOD: Try to find the market by question or other fields
            if isinstance(market_id, (int, float)):
                market_id_str = str(market_id)
                # Try exact string match on ID (handles format differences)
                str_matches = main_df[main_df['id'].astype(str) == market_id_str]
                if not str_matches.empty:
                    token_ids_str = str_matches.iloc[0]['clobTokenIds']
                    try:
                        token_ids = json.loads(token_ids_str)
                    except json.JSONDecodeError:
                        import ast
                        token_ids = ast.literal_eval(token_ids_str)
                    return token_ids
            
            # Try by name/question if ID wasn't found
            if 'question' in main_df.columns:
                # Try to find the market question from the ID
                question = None
                try:
                    # Try to load market question mapping
                    with open('data/trades/market_id_to_question.json', 'r') as f:
                        mapping = json.load(f)
                    question = mapping.get(str(market_id))
                except Exception:
                    pass
                
                if question:
                    # Find the market by question
                    question_matches = main_df[main_df['question'] == question]
                    if not question_matches.empty:
                        token_ids_str = question_matches.iloc[0]['clobTokenIds']
                        try:
                            token_ids = json.loads(token_ids_str)
                        except json.JSONDecodeError:
                            import ast
                            token_ids = ast.literal_eval(token_ids_str)
                        return token_ids
            
            print(f"No market found with ID: {market_id}")
            return []
    
    except Exception as e:
        print(f"Error getting token IDs for market {market_id}: {e}")
        return []

def find_token_id_file(token_id: str, trades_dir='data/trades/trades') -> Optional[str]:
    """
    Find a parquet file that matches a given token ID
    
    Args:
        token_id: The token ID to search for
        trades_dir: Directory containing trade data files
    
    Returns:
        Path to the matching file or None if no match found
    """
    if not os.path.exists(trades_dir):
        print(f"Trades directory does not exist: {trades_dir}")
        return None
        
    # First try exact match
    exact_match = os.path.join(trades_dir, f"{token_id}.parquet")
    if os.path.exists(exact_match):
        return exact_match
    
    # Try prefix match (files might have timestamps or other suffixes)
    for filename in os.listdir(trades_dir):
        if str(token_id) in filename and filename.endswith('.parquet'):
            return os.path.join(trades_dir, filename)
    
    # If no match found
    return None

def load_trade_data(market_id: Union[str, int, float], 
                    trades_dir='data/trades', 
                    token_type='yes',  # Add parameter to specify token type
                    return_all_tokens=False) -> Optional[pd.DataFrame]:  # Change default to False
    """
    Load trade data for a market with improved reliability
    
    Args:
        market_id: Market ID to load trade data for
        trades_dir: Base directory containing trade data
        token_type: Type of token to load ('yes', 'no', or 'all')
        return_all_tokens: Whether to return all tokens or just the specified type
        
    Returns:
        DataFrame with trade data or None if not found
    """
    # Get token IDs for this market
    token_ids = get_token_ids_for_market(market_id)
    
    if not token_ids:
        print(f"No token IDs found for market {market_id}")
        return None
    
    # List to store DataFrames for each token
    all_token_dfs = []
    token_info = {}
    
    # Try to load market question mapping to identify token types
    try:
        market_tokens = load_market_tokens('data/trades/market_tokens.json')
        question = None
        
        # Get the market question
        for q, tokens in market_tokens.items():
            if any(str(tid) in tokens for tid in token_ids):
                question = q
                break
        
        if question:
            # Determine which token is "Yes" and which is "No"
            for i, tid in enumerate(token_ids):
                # Simple heuristic: first token is typically "Yes"
                # This should be improved with actual metadata if available
                token_info[tid] = "yes" if i == 0 else "no"
    except Exception as e:
        print(f"Warning: Could not determine token types: {e}")
        # Fall back to simple heuristic
        for i, tid in enumerate(token_ids):
            token_info[tid] = "yes" if i == 0 else "no"
    
    # Load data for each token
    for token_id in token_ids:
        # Skip tokens that don't match the requested type
        if token_type != 'all' and token_info.get(token_id, '') != token_type and not return_all_tokens:
            continue
            
        # Look for the token file
        token_file = find_token_id_file(token_id, os.path.join(trades_dir, 'trades'))
        
        if token_file and os.path.exists(token_file):
            try:
                # Load the parquet file
                df = pq.read_table(token_file).to_pandas()
                
                # Add token information
                df['market_id'] = market_id
                df['token_id'] = token_id
                df['token_type'] = token_info.get(token_id, 'unknown')
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
                
                # Sort by timestamp
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                
                all_token_dfs.append(df)
                print(f"Loaded {len(df)} trades for token {token_id} ({token_info.get(token_id, 'unknown')})")
            except Exception as e:
                print(f"Error loading file for token {token_id}: {e}")
    
    # Rest of function remains the same...
    
    if not all_token_dfs:
        print(f"No trade data found for market {market_id}")
        return None
    
    # Return data based on user preference
    if return_all_tokens:
        combined_df = pd.concat(all_token_dfs, ignore_index=True)
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp')
        return combined_df
    else:
        # Filter for the requested token type if multiple were loaded
        if token_type != 'all' and len(all_token_dfs) > 1:
            yes_tokens = [df for df in all_token_dfs if df['token_type'].iloc[0] == token_type]
            if yes_tokens:
                return yes_tokens[0]
        
        # If no specific filtering or only one token loaded, return the first DataFrame
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