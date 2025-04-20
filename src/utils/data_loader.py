#!/usr/bin/env python3
'''
Data Loader

This module will contain code for analyzing Polymarket data focusing on data loader.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from imblearn.over_sampling import SMOTE

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_main_dataset(filepath='data/cleaned_election_data.csv'):
    """
    Load the main dataset with market features
    """
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def load_market_question_mapping(filepath='data/trades/market_id_to_question.json'):
    """
    Load mapping between market IDs and questions
    """
    with open(filepath, 'r') as f:
        mapping = json.load(f)
    return mapping

def load_trade_data(market_id, trades_dir='data/trades'):
    """
    Load trade-level data for a specific market
    """
    filepath = os.path.join(trades_dir, f"{market_id}.parquet")
    if not os.path.exists(filepath):
        print(f"No trade data found for market {market_id}")
        return None
    
    trades_df = pd.read_parquet(filepath)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp')
    return trades_df

def summarize_dataset(df):
    """
    Print summary statistics and information about the dataset
    """
    print("Dataset summary:")
    print(f"Number of markets: {df.shape[0]}")
    print(f"Prediction accuracy: {df['prediction_correct'].mean()*100:.2f}%")
    print(f"Average Brier score: {df['brier_score'].mean():.4f}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\nColumns with missing values:")
    print(missing[missing > 0].sort_values(ascending=False))
    
    # Print summary of key features
    print("\nSummary of key features:")
    key_features = ['brier_score', 'log_loss', 'price_range', 'price_volatility', 
                    'final_week_momentum', 'volumeNum', 'unique_traders_count', 
                    'trader_concentration', 'buy_sell_ratio']
    
    print(df[key_features].describe())
    
    # Distribution of markets by election type and region
    print("\nDistribution by election type:")
    print(df['event_electionType'].value_counts())
    
    print("\nDistribution by country:")
    print(df['event_country'].value_counts().head(10))