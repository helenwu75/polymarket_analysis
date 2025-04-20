import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def explore_data_structure(base_path="data"):
    """
    Explore the data structure in the polymarket_analysis repository
    
    Args:
        base_path: Base directory containing the data files
    """
    print(f"Exploring data structure in {base_path}")
    
    # Check main directories
    directories = ["trades", "external"]
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        if os.path.exists(dir_path):
            print(f"\nDirectory {directory}:")
            files = os.listdir(dir_path)
            if files:
                print(f"  Contains {len(files)} files/directories")
                # Show a sample of files if there are many
                if len(files) > 5:
                    print(f"  Sample files: {files[:5]}")
                else:
                    print(f"  Files: {files}")
            else:
                print("  Empty directory")
    
    # Examine cleaned_election_data.csv
    main_data_file = os.path.join(base_path, "cleaned_election_data.csv")
    if os.path.exists(main_data_file):
        print(f"\nExamining main data file: {main_data_file}")
        try:
            df = pd.read_csv(main_data_file)
            print(f"Shape: {df.shape} (rows, columns)")
            print(f"Columns: {df.columns.tolist()}")
            print("\nSample data (first 3 rows):")
            print(df.head(3))
            
            # Get data types and check for missing values
            print("\nData types:")
            print(df.dtypes)
            
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("\nColumns with missing values:")
                print(missing[missing > 0])
            else:
                print("\nNo missing values found")
            
            # Check for target variables
            target_vars = ['brier_score', 'log_loss', 'prediction_correct']
            available_targets = [var for var in target_vars if var in df.columns]
            if available_targets:
                print(f"\nTarget variables found: {available_targets}")
                for target in available_targets:
                    print(f"{target}: min={df[target].min()}, max={df[target].max()}, mean={df[target].mean():.4f}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
    else:
        print(f"Main data file not found: {main_data_file}")
    
    # Check trades directory structure
    trades_dir = os.path.join(base_path, "trades")
    if os.path.exists(trades_dir):
        print(f"\nExploring trades directory structure:")
        
        # List subdirectories
        subdirs = [d for d in os.listdir(trades_dir) if os.path.isdir(os.path.join(trades_dir, d))]
        if subdirs:
            print(f"Found {len(subdirs)} market subdirectories")
            
            # Examine one sample subdirectory
            sample_dir = os.path.join(trades_dir, subdirs[0])
            print(f"\nExamining sample market directory: {subdirs[0]}")
            
            # Check files in sample directory
            files = os.listdir(sample_dir)
            print(f"Files in sample directory: {files}")
            
            # Examine a sample trade file if available
            trade_files = [f for f in files if f.endswith('.csv') or f.endswith('.parquet')]
            if trade_files:
                sample_file = os.path.join(sample_dir, trade_files[0])
                print(f"\nExamining sample trade file: {trade_files[0]}")
                
                try:
                    # Try to read file based on extension
                    if sample_file.endswith('.csv'):
                        trade_data = pd.read_csv(sample_file)
                    elif sample_file.endswith('.parquet'):
                        trade_data = pd.read_parquet(sample_file)
                    
                    print(f"Shape: {trade_data.shape}")
                    print(f"Columns: {trade_data.columns.tolist()}")
                    print("\nSample trade data (first 3 rows):")
                    print(trade_data.head(3))
                except Exception as e:
                    print(f"Error reading trade file: {e}")
        else:
            print("No market subdirectories found")

if __name__ == "__main__":
    print("Data Structure Analysis for Polymarket Prediction Project")
    print("=" * 60)
    
    # Explore the data structure
    explore_data_structure()
    
    print("\nAnalysis complete!")