# run_market_efficiency.py
# batch analysis: python run_market_efficiency.py --market_ids 253591 253597 253642 --verbose
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for importing utilities
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir) if current_dir.endswith('notebooks') else current_dir
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility functions and analyzer
try:
    from src.utils.data_loader import load_main_dataset, load_trade_data
    print("Successfully imported data_loader utilities")
except ImportError:
    print("Warning: Could not import data_loader utilities. Some functions may not work.")
    from src.knowledge_value.market_efficiency import MarketEfficiencyAnalyzer
else:
    from src.knowledge_value.market_efficiency import MarketEfficiencyAnalyzer

def json_serializable(obj):
    """Convert non-serializable objects to serializable ones for JSON"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    # Add this line to handle any other non-serializable objects
    return str(obj)

def run_efficiency_analysis(market_selection, results_dir='results/market_efficiency', verbose=True):
    """
    Run market efficiency analysis on selected markets
    
    Parameters:
    -----------
    market_selection : dict
        Dictionary with market selection parameters
    results_dir : str
        Directory to save results
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    tuple
        (results_list, summary) - List of analysis results and summary statistics
    """
    # Create directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"analysis_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    if verbose:
        print(f"Results will be saved to: {run_dir}")
    
    # Initialize analyzer
    analyzer = MarketEfficiencyAnalyzer(results_dir=run_dir)
    
    # Load market data
    if verbose:
        print("\nLoading main dataset...")
    market_data = load_main_dataset('data/cleaned_election_data.csv')
    
    if market_data is None or market_data.empty:
        print("Error: Failed to load market data")
        return [], {}
    
    # Select markets to analyze
    selected_markets = select_markets(market_data, market_selection)
    
    if selected_markets is None or len(selected_markets) == 0:
        print("Error: No markets selected for analysis")
        return [], {}
    
    if verbose:
        print(f"\nSelected {len(selected_markets)} markets for analysis:")
        for i, (idx, row) in enumerate(selected_markets.iterrows()):
            market_name = row['question'] if 'question' in row else f"Market {row['id']}"
            print(f"  {i+1}. {market_name} (ID: {row['id']})")
    
    # Analyze each market
    results_list = []
    
    for i, (idx, row) in enumerate(tqdm(selected_markets.iterrows(), desc="Analyzing markets", 
                                       total=len(selected_markets), disable=not verbose)):
        market_id = row['id']
        market_name = row['question'] if 'question' in row else f"Market {market_id}"
        
        # Load trade data
        if verbose:
            print(f"\nAnalyzing market: {market_name} (ID: {market_id})")
            print("Loading trade data...")
        
        trade_data = load_trade_data(market_id)
        
        if trade_data is None or len(trade_data) < 30:
            if verbose:
                print(f"Insufficient trade data for market {market_id}")
            results_list.append({
                'market_id': market_id,
                'market_name': market_name,
                'analysis_success': False,
                'reason': 'Insufficient trade data'
            })
            continue
        
        # Preprocess data
        if verbose:
            print(f"Preprocessing trade data ({len(trade_data)} trades)...")
        
        processed_data = analyzer.preprocess_market_data(trade_data)
        
        if processed_data is None or len(processed_data) < 30:
            if verbose:
                print(f"Failed to preprocess data for market {market_id}")
            results_list.append({
                'market_id': market_id,
                'market_name': market_name,
                'analysis_success': False,
                'reason': 'Preprocessing failed'
            })
            continue
        
        # Run analysis
        if verbose:
            print(f"Running efficiency tests...")
        
        result = analyzer.analyze_market(processed_data, market_id, market_name)
        
        # Save result
        results_list.append(result)
        
        # Create visualization
        if result['analysis_success']:
            if verbose:
                print(f"Creating visualization...")
            
            # Create safe filename
            safe_name = ''.join(c if c.isalnum() else '_' for c in market_name)[:50]
            viz_path = os.path.join(run_dir, f"market_{market_id}_{safe_name}.png")
            
            analyzer.visualize_market(processed_data, result, market_name, viz_path)
    
    # Create comparison visualization if multiple markets analyzed
    if len(results_list) > 1:
        if verbose:
            print("\nCreating comparison visualization...")
        
        successful = [r for r in results_list if r.get('analysis_success', False)]
        
        if successful:
            comparison_path = os.path.join(run_dir, "market_comparison.png")
            analyzer.visualize_comparison(successful, comparison_path)
    
    # Generate summary
    if verbose:
        print("\nGenerating summary report...")
    
    summary = analyzer.generate_summary_report(results_list)
    
    # Save results to files
    if verbose:
        print("\nSaving results...")
    
    # Save individual results
    with open(os.path.join(run_dir, "market_results.json"), 'w') as f:
        json.dump(results_list, f, indent=2, default=json_serializable)
    
    # Save summary
    with open(os.path.join(run_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=json_serializable)
    
    # Print summary statistics
    if verbose and 'total_markets' in summary:
        print("\n" + "="*50)
        print("MARKET EFFICIENCY ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total markets analyzed: {summary['total_markets']}")
        print(f"Average efficiency score: {summary['avg_efficiency_score']:.2f}/100")
        print(f"Median efficiency score: {summary['median_efficiency_score']:.2f}/100")
        
        print("\nEfficiency Classification:")
        for cls, count in summary['classifications'].items():
            print(f"  {cls}: {count} markets ({count/summary['total_markets']*100:.1f}%)")
        
        print("\nTest Results:")
        for test, percentage in summary['test_percentages'].items():
            print(f"  {test.replace('_', ' ').title()}: {percentage:.1f}%")
        
        print("="*50)
    

    # Generate aggregate report
    if verbose:
        print("\nGenerating aggregate report...")

    aggregate_report = [
        "# Market Efficiency Analysis: Aggregate Report",
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- **Total Markets Analyzed**: {summary['total_markets']}",
        f"- **Average Efficiency Score**: {summary['avg_efficiency_score']:.2f}/100",
        f"- **Median Efficiency Score**: {summary['median_efficiency_score']:.2f}/100",
        f"- **Standard Deviation**: {summary['std_efficiency_score']:.2f}",
        f"- **Range**: {summary['min_efficiency_score']:.2f} - {summary['max_efficiency_score']:.2f}",
        "",
        "## Efficiency Classifications",
    ]

    # Add classification breakdown
    for cls, count in summary['classifications'].items():
        percentage = count / summary['total_markets'] * 100
        aggregate_report.append(f"- **{cls}**: {count} markets ({percentage:.1f}%)")

    aggregate_report.extend([
        "",
        "## Test Results",
    ])

    # Add test results
    for test, percentage in summary['test_percentages'].items():
        test_name = test.replace('_', ' ').title()
        aggregate_report.append(f"- **{test_name}**: {percentage:.1f}% of markets passed")

    aggregate_report.extend([
        "",
        "## Analyzed Markets",
    ])

    # Add list of analyzed markets
    for i, result in enumerate(results_list):
        if result['analysis_success']:
            market_name = result.get('market_name', f"Market {result.get('market_id', i)}")
            market_id = result.get('market_id', "Unknown")
            efficiency_score = result.get('efficiency_score', 0)
            efficiency_class = result.get('efficiency_class', "Unknown")
            aggregate_report.append(f"- **{market_name}** (ID: {market_id}): {efficiency_score:.2f}/100 - {efficiency_class}")

    # Add conclusions
    aggregate_report.extend([
        "",
        "## Conclusion",
    ])

    avg_score = summary['avg_efficiency_score']
    if avg_score < 40:
        aggregate_report.append("The majority of analyzed markets exhibit significant inefficiencies, with strong evidence against the random walk hypothesis across multiple tests. This suggests that these prediction markets frequently display predictable patterns that could potentially be exploited by informed traders.")
    elif avg_score < 60:
        aggregate_report.append("The analyzed markets show mixed efficiency results. Some markets follow random walk patterns while others display predictable behaviors. This suggests varying levels of information incorporation across different prediction markets.")
    elif avg_score < 80:
        aggregate_report.append("Most analyzed markets demonstrate moderate efficiency, with the majority of tests supporting the random walk hypothesis. While some inefficiencies exist, prediction markets generally appear to incorporate new information reasonably well.")
    else:
        aggregate_report.append("The analyzed prediction markets are largely efficient, strongly supporting the random walk hypothesis. Price movements typically appear to be unpredictable, suggesting that these markets effectively incorporate available information.")

    # Save aggregate report
    aggregate_report_path = os.path.join(run_dir, "aggregate_report.md")
    with open(aggregate_report_path, 'w') as f:
        f.write('\n'.join(aggregate_report))

    if verbose:
        print(f"Aggregate report saved to: {aggregate_report_path}")
    return results_list, summary

def select_markets(market_data, market_selection):
    """
    Select markets to analyze based on selection criteria
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        DataFrame with market data
    market_selection : dict
        Dictionary with market selection parameters
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with selected markets
    """
    # Initialize empty DataFrame for selections
    selected_markets = pd.DataFrame()
    use_default = True  # Flag to determine if we need to fall back to defaults
    
    # Filter by name if specified
    if market_selection.get('by_name') and len(market_selection['by_name']) > 0:
        # Fix any typos in market names
        fixed_names = [name.replace("New Meixco", "New Mexico") for name in market_selection['by_name']]
        name_matches = market_data[market_data['question'].isin(fixed_names)]
        
        if len(name_matches) > 0:
            selected_markets = pd.concat([selected_markets, name_matches]).drop_duplicates()
            use_default = False  # We found some matches, don't use default
            print(f"Selected {len(name_matches)} markets by name")
    
    # Filter by ID if specified
    if market_selection.get('by_id') and len(market_selection['by_id']) > 0:
        id_filter = market_data['id'].isin(market_selection['by_id'])
        id_matches = market_data[id_filter]
        
        if len(id_matches) > 0:
            selected_markets = pd.concat([selected_markets, id_matches]).drop_duplicates()
            use_default = False  # We found some matches, don't use default
            print(f"Selected {len(id_matches)} markets by ID")
    
    # Only add top markets by volume if explicitly requested OR if we need defaults
    if use_default or market_selection.get('top_n_by_volume', 0) > 0:
        if 'volumeNum' in market_data.columns:
            top_count = market_selection.get('top_n_by_volume', 5) if market_selection.get('top_n_by_volume', 0) > 0 else 5
            top_markets = market_data.sort_values('volumeNum', ascending=False).head(top_count)
            
            if len(top_markets) > 0:
                selected_markets = pd.concat([selected_markets, top_markets]).drop_duplicates()
                print(f"Selected {len(top_markets)} top markets by volume")
    
    # Apply minimum volume filter if specified
    if market_selection.get('min_volume', 0) > 0 and 'volumeNum' in market_data.columns and len(selected_markets) > 0:
        volume_filter = selected_markets['volumeNum'] >= market_selection['min_volume']
        filtered_markets = selected_markets[volume_filter]
        
        # Only apply filter if it doesn't remove all markets
        if len(filtered_markets) > 0:
            selected_markets = filtered_markets
            print(f"Selected {len(selected_markets)} markets with minimum volume {market_selection['min_volume']}")
    
    # Apply date range filter if specified
    if market_selection.get('date_range') and 'market_start_date' in market_data.columns:
        start_date, end_date = market_selection['date_range']
        
        # Convert dates to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(market_data['market_start_date']):
            market_data['market_start_date'] = pd.to_datetime(market_data['market_start_date'])
        
        date_filter = (market_data['market_start_date'] >= start_date) & (market_data['market_start_date'] <= end_date)
        date_matches = market_data[date_filter]
        
        if len(date_matches) > 0:
            selected_markets = pd.concat([selected_markets, date_matches]).drop_duplicates()
            print(f"Selected {len(date_matches)} markets in date range {start_date} to {end_date}")
    
    print(f"Final selection: {len(selected_markets)} markets")

    
    return selected_markets

def main():
    """Main function to run market efficiency analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run market efficiency analysis on Polymarket data")
    parser.add_argument("--data_dir", default="data", help="Directory with the market data")
    parser.add_argument("--results_dir", default="results/market_efficiency", help="Directory to save results")
    parser.add_argument("--market_ids", nargs='+', help="List of market IDs to analyze")
    parser.add_argument("--market_names", nargs='+', help="List of market names to analyze")
    parser.add_argument("--top_n", type=int, default=0, help="Analyze top N markets by volume")
    parser.add_argument("--min_volume", type=float, default=0, help="Minimum volume threshold")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    # Configure market selection
    market_selection = {
        'by_name': args.market_names if args.market_names else [],
        'by_id': [int(mid) for mid in args.market_ids] if args.market_ids else [],
        'top_n_by_volume': args.top_n,
        'min_volume': args.min_volume,
        'date_range': None
    }
    
    # Run analysis
    run_efficiency_analysis(market_selection, args.results_dir, args.verbose)

if __name__ == "__main__":
    main()