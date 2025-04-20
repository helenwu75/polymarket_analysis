#!/usr/bin/env python3
"""
Test script for the modified data loader functions
"""

import os
import sys
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the modified functions
from utils.data_loader import (
    load_main_dataset,
    load_market_question_mapping,
    load_market_tokens,
    load_trade_data,
    get_sample_market_ids,
    summarize_dataset
)

def test_data_loading():
    """Test that the data loading functions work correctly"""
    print("=== Testing Data Loading Functions ===\n")
    
    # 1. Test loading the main dataset
    print("1. Loading main dataset...")
    df = load_main_dataset()
    print(f"Successfully loaded main dataset with shape: {df.shape}")
    
    # 2. Test loading market-question mapping
    print("\n2. Loading market-question mapping...")
    mapping = load_market_question_mapping()
    print(f"Successfully loaded mapping with {len(mapping)} entries")
    
    # 3. Test loading market tokens (if file exists)
    print("\n3. Loading market tokens...")
    try:
        tokens = load_market_tokens()
        print(f"Successfully loaded market tokens with {len(tokens)} entries")
        # Show a sample token ID
        if tokens:
            market_id = next(iter(tokens))
            token_ids = tokens[market_id]
            print(f"Sample market ID: {market_id}")
            print(f"Associated token IDs: {token_ids}")
    except Exception as e:
        print(f"Error loading market tokens: {e}")
    
    # 4. Get sample market IDs
    print("\n4. Getting sample market IDs...")
    market_ids = get_sample_market_ids(3)
    print(f"Sample market IDs: {market_ids}")
    
    # 5. Test loading trade data for each sample market ID
    print("\n5. Testing trade data loading for sample markets...")
    for i, market_id in enumerate(market_ids):
        print(f"\nMarket {i+1}: {market_id}")
        print(f"- Question: {mapping.get(market_id, 'Unknown')}")
        
        # Load trade data
        trades_df = load_trade_data(market_id)
        if trades_df is not None:
            print(f"- Successfully loaded trade data with shape: {trades_df.shape}")
            print(f"- Time range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
            print(f"- Sample columns: {trades_df.columns.tolist()}")
            print(f"- First row: {trades_df.iloc[0][['price', 'side', 'size', 'timestamp']].to_dict()}")
        else:
            print("- No trade data found for this market")
    
    print("\nData loading test completed!")

if __name__ == "__main__":
    test_data_loading()