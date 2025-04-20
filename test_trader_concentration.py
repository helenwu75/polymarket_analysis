#!/usr/bin/env python3
"""
Simple test script to verify trader concentration analysis
"""

import os
import sys
import matplotlib.pyplot as plt
# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing necessary modules
try:
    from src.utils.data_loader import load_main_dataset, load_market_question_mapping
    from src.cultural_value.trader_concentration import calculate_trader_concentration, plot_trader_concentration
    print("Successfully imported required modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Load dataset and mapping
try:
    df = load_main_dataset()
    mapping = load_market_question_mapping()
    print(f"Successfully loaded main dataset with shape: {df.shape}")
    print(f"Successfully loaded market-question mapping with {len(mapping)} entries")
except Exception as e:
    print(f"Error loading initial data: {e}")
    sys.exit(1)

# Select a sample market for analysis
sample_market_id = None
try:
    # Try to find a market with non-null trader_concentration
    markets_with_concentration = df[~df['trader_concentration'].isna()]
    if not markets_with_concentration.empty:
        sample_market_id = markets_with_concentration.iloc[0]['market_id']
        print(f"Selected market ID with known trader concentration: {sample_market_id}")
    else:
        # Just take the first market ID
        sample_market_id = df.iloc[0]['market_id']
        print(f"Selected first market ID: {sample_market_id}")
except Exception as e:
    print(f"Error selecting sample market: {e}")
    # If market selection failed, exit
    if sample_market_id is None:
        print("No market selected, exiting test.")
        sys.exit(1)

# Test calculating trader concentration
print(f"\nCalculating trader concentration for market {sample_market_id}...")
try:
    concentration_results, trader_volumes = calculate_trader_concentration(sample_market_id)
    if concentration_results:
        print("Successfully calculated trader concentration:")
        for key, value in concentration_results.items():
            print(f"  {key}: {value}")
    else:
        print("No concentration results returned")
except Exception as e:
    print(f"Error calculating trader concentration: {e}")

# Test plotting trader concentration
if 'trader_volumes' in locals() and trader_volumes is not None:
    print("\nPlotting trader concentration...")
    try:
        # Disable actual display for testing
        plt.ioff()
        plot_trader_concentration(trader_volumes, sample_market_id, mapping)
        plt.close()
        print("Successfully created trader concentration plot")
    except Exception as e:
        print(f"Error plotting trader concentration: {e}")
else:
    print("Skipping plot as trader volumes data is not available")

print("\nTest completed!")