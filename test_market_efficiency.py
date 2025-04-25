# test_market_efficiency.py
# usage: python test_market_efficiency.py 253591 --verbose

import os
import sys
try:
    from src.utils.data_loader import load_main_dataset, load_trade_data
    print("Successfully imported data_loader utilities")
except ImportError:
    print("Warning: Could not import data_loader utilities. Some functions may not work.")
    from src.knowledge_value.market_efficiency import MarketEfficiencyAnalyzer
else:
    from src.knowledge_value.market_efficiency import MarketEfficiencyAnalyzer

def test_market(market_id, data_dir='data', results_dir='results/market_efficiency', verbose=True):
    """
    Test the efficiency of a specific market
    
    Parameters:
    -----------
    market_id : int or str
        ID of the market to test
    data_dir : str
        Directory with the market data
    results_dir : str
        Directory to save results
    verbose : bool
        Whether to print verbose output
    """
    # Add parent directory to path for importing utilities
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir) if current_dir.endswith('notebooks') else current_dir
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Import utility functions
    try:
        from src.utils.data_loader import load_main_dataset, load_trade_data
    except ImportError:
        print("Error: Could not import data_loader utilities.")
        return
    
    # Create directory for results
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MarketEfficiencyAnalyzer(results_dir=results_dir)
    
    # Load market data to get name
    market_name = None
    try:
        market_data = load_main_dataset(os.path.join(data_dir, 'cleaned_election_data.csv'))
        if market_data is not None:
            market_row = market_data[market_data['id'] == int(market_id)]
            if not market_row.empty and 'question' in market_row.columns:
                market_name = market_row.iloc[0]['question']
    except:
        pass
    
    if market_name is None:
        market_name = f"Market {market_id}"
    
    if verbose:
        print(f"Testing efficiency for market: {market_name} (ID: {market_id})")
    
    # Load trade data
    if verbose:
        print("Loading trade data...")
    
    trade_data = load_trade_data(market_id)
    
    if trade_data is None or len(trade_data) < 30:
        print(f"Error: Insufficient trade data for market {market_id}")
        return
    
    if verbose:
        print(f"Loaded {len(trade_data)} trades")
    
    # Preprocess data
    if verbose:
        print("Preprocessing trade data...")
    
    processed_data = analyzer.preprocess_market_data(trade_data)
    
    if processed_data is None or len(processed_data) < 30:
        print("Error: Failed to preprocess data or insufficient time points")
        return
    
    if verbose:
        print(f"Successfully preprocessed data with {len(processed_data)} time points")
    
    # Run analysis
    if verbose:
        print("Running efficiency tests...")
    
    result = analyzer.analyze_market(processed_data, market_id, market_name)
    
    if not result['analysis_success']:
        print(f"Error: Analysis failed - {result.get('reason', 'Unknown reason')}")
        return
    
    # Print results
    print("\n" + "="*50)
    print(f"MARKET EFFICIENCY RESULTS: {market_name}")
    print("="*50)
    print(f"Efficiency Score: {result['efficiency_score']:.2f}/100")
    print(f"Classification: {result['efficiency_class']}")
    
    print("\nTest Results:")
    
    if 'adf_price' in result and result['adf_price']:
        is_random_walk = not result['adf_price']['is_stationary']
        print(f"Random Walk Test: {'Pass' if is_random_walk else 'Fail'} (p-value: {result['adf_price']['p_value']:.4f})")
    
    if 'adf_return' in result and result['adf_return']:
        is_stationary = result['adf_return']['is_stationary']
        print(f"Return Stationarity Test: {'Pass' if is_stationary else 'Fail'} (p-value: {result['adf_return']['p_value']:.4f})")
    
    if 'autocorrelation' in result and result['autocorrelation']:
        no_autocorr = not result['autocorrelation']['has_significant_autocorrelation']
        print(f"No Autocorrelation Test: {'Pass' if no_autocorr else 'Fail'}")
        
        if not no_autocorr:
            print(f"  Significant lags: {result['autocorrelation']['significant_lags']}")
    
    if 'runs_test' in result and result['runs_test']:
        is_random = result['runs_test']['is_random']
        print(f"Runs Test for Randomness: {'Pass' if is_random else 'Fail'} (p-value: {result['runs_test']['p_value']:.4f})")
    
    if 'ar_model' in result and result['ar_model']:
        not_predictable = not result['ar_model']['is_significant']
        print(f"AR Model Test: {'Pass' if not_predictable else 'Fail'} (p-value: {result['ar_model']['p_value']:.4f})")
    
    print("="*50)
    
    # Create visualization
    if verbose:
        print("Creating visualization...")
    
    # Create safe filename
    safe_name = ''.join(c if c.isalnum() else '_' for c in market_name)[:50]
    viz_path = os.path.join(results_dir, f"market_{market_id}_{safe_name}.png")
    
    analyzer.visualize_market(processed_data, result, market_name, viz_path)
    
    if verbose:
        print(f"Visualization saved to: {viz_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test market efficiency for a specific market")
    parser.add_argument("market_id", help="ID of the market to test")
    parser.add_argument("--data_dir", default="data", help="Directory with the market data")
    parser.add_argument("--results_dir", default="results/market_efficiency", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    test_market(args.market_id, args.data_dir, args.results_dir, args.verbose)