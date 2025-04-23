# analyze_efficiency.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import sys

# Add the project path
if 'src' not in sys.path:
    sys.path.append('src')

from market_efficiency_analysis import MarketEfficiencyAnalyzer
from strong_form_efficiency import StrongFormEfficiencyAnalyzer

def run_comprehensive_analysis(data_dir='data', results_dir='results/market_efficiency', 
                              market_count=50, verbose=True):
    """
    Run a comprehensive analysis of market efficiency for Polymarket data
    
    Parameters:
    -----------
    data_dir : str
        Directory with the data
    results_dir : str
        Directory to save results
    market_count : int
        Maximum number of markets to analyze
    verbose : bool
        Whether to print verbose output
    
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize analyzers
    analyzer = MarketEfficiencyAnalyzer(data_dir=data_dir, results_dir=results_dir)
    strong_form_analyzer = StrongFormEfficiencyAnalyzer(analyzer)
    
    # Check if data was loaded successfully
    if analyzer.main_df.empty:
        print("Error: Failed to load main dataset")
        return None
    
    results = {
        'dataset_info': {
            'total_markets': len(analyzer.main_df),
            'markets_analyzed': 0
        },
        'weak_form': {},
        'time_varying': {},
        'strong_form': {},
        'aggregate': {}
    }
    
    if verbose:
        print(f"Loaded {len(analyzer.main_df)} markets")
    
    # 1. Find markets by category
    market_categories = {
        'presidential': analyzer.find_market_by_name("Presidential"),
        'senate': analyzer.find_market_by_name("Senate"),
        'parliamentary': analyzer.find_market_by_name("Parliamentary"),
        'governor': analyzer.find_market_by_name("Governor")
    }
    
    if verbose:
        for category, markets in market_categories.items():
            print(f"Found {len(markets)} {category} markets")
    
    # 2. Weak-form efficiency testing
    weak_form_results = []
    
    # Get markets to analyze
    markets_to_analyze = []
    for category, markets in market_categories.items():
        if markets:
            # Take up to 5 markets from each category
            category_markets = markets[:min(5, len(markets))]
            markets_to_analyze.extend([m[0] for m in category_markets])
    
    # Limit to market_count
    if len(markets_to_analyze) > market_count:
        markets_to_analyze = markets_to_analyze[:market_count]
    
    # Run analysis
    if markets_to_analyze:
        if verbose:
            print(f"\nRunning weak-form efficiency tests on {len(markets_to_analyze)} markets...")
        
        for market_id in tqdm(markets_to_analyze, desc="Analyzing markets"):
            result = analyzer.analyze_market(market_id, verbose=False)
            if result.get('analysis_success', False):
                weak_form_results.append(result)
        
        results['dataset_info']['markets_analyzed'] = len(weak_form_results)
        results['weak_form']['results'] = weak_form_results
        
        # Calculate aggregate statistics
        if weak_form_results:
            efficiency_scores = [r['efficiency_score'] for r in weak_form_results]
            efficiency_classes = {}
            
            for result in weak_form_results:
                if 'efficiency_class' in result:
                    cls = result['efficiency_class']
                    efficiency_classes[cls] = efficiency_classes.get(cls, 0) + 1
            
            results['aggregate']['avg_efficiency_score'] = np.mean(efficiency_scores)
            results['aggregate']['median_efficiency_score'] = np.median(efficiency_scores)
            results['aggregate']['std_efficiency_score'] = np.std(efficiency_scores)
            results['aggregate']['efficiency_classes'] = efficiency_classes
    else:
        if verbose:
            print("No markets found for weak-form efficiency testing")
    
    # 3. Time-varying efficiency analysis
    time_varying_markets = []
    
    # Select markets with enough data
    for market_id in markets_to_analyze:
        market_data = analyzer.preprocess_market_data(market_id)
        if market_data is not None and len(market_data) >= 90:
            time_varying_markets.append(market_id)
    
    time_varying_results = []
    
    if time_varying_markets:
        if verbose:
            print(f"\nRunning time-varying efficiency analysis on {len(time_varying_markets)} markets...")
        
        for market_id in tqdm(time_varying_markets, desc="Analyzing time-varying efficiency"):
            market_data = analyzer.preprocess_market_data(market_id)
            tv_result = analyzer.analyze_time_varying_efficiency(market_data['log_return'])
            
            if tv_result and 'summary' in tv_result:
                # Add market info
                market_info = analyzer.get_market_details(market_id)
                tv_result['market_id'] = market_id
                tv_result['market_name'] = market_info['question']
                
                time_varying_results.append(tv_result)
        
        results['time_varying']['results'] = time_varying_results
        
        # Calculate aggregate statistics
        if time_varying_results:
            efficiency_changes = {
                'More Efficient': 0,
                'Less Efficient': 0,
                'No Change': 0
            }
            
            volatility_changes = []
            
            for result in time_varying_results:
                if 'summary' in result:
                    change = result['summary']['efficiency_change']
                    efficiency_changes[change] = efficiency_changes.get(change, 0) + 1
                    
                    if 'volatility_change' in result['summary']:
                        volatility_changes.append(result['summary']['volatility_change'])
            
            results['time_varying']['efficiency_changes'] = efficiency_changes
            results['time_varying']['avg_volatility_change'] = np.mean(volatility_changes) if volatility_changes else None
    else:
        if verbose:
            print("No markets found with sufficient data for time-varying analysis")
    
    # 4. Strong-form efficiency (event study)
    event_study_markets = []
    
    # Find markets with potential events
    for market_id in markets_to_analyze:
        significant_events = strong_form_analyzer.identify_significant_events(market_id)
        if significant_events is not None and len(significant_events) >= 3:
            event_study_markets.append((market_id, significant_events))
    
    event_study_results = []
    
    if event_study_markets:
        if verbose:
            print(f"\nRunning event studies on {len(event_study_markets)} markets...")
        
        for market_id, events in tqdm(event_study_markets, desc="Running event studies"):
            event_dates = events.index.tolist()
            study_result = strong_form_analyzer.run_event_study(market_id, event_dates)
            
            if study_result and len(study_result['event_results']) > 0:
                # Add market info
                market_info = analyzer.get_market_details(market_id)
                study_result['market_name'] = market_info['question']
                
                event_study_results.append(study_result)
                
                # Create visualization
                fig = strong_form_analyzer.visualize_event_study(study_result)
                if fig:
                    market_name = market_info['question']
                    safe_name = "".join(c if c.isalnum() else "_" for c in market_name[:30])
                    plt.savefig(os.path.join(results_dir, f"event_study_{market_id}_{safe_name}.png"), dpi=300)
                    plt.close(fig)
        
        results['strong_form']['results'] = event_study_results
        
        # Calculate aggregate statistics
        if event_study_results:
            speed_of_adjustment = []
            volatility_ratios = []
            
            for result in event_study_results:
                if 'avg_metrics' in result:
                    metrics = result['avg_metrics']
                    if 'avg_speed_of_adjustment' in metrics:
                        speed_of_adjustment.append(metrics['avg_speed_of_adjustment'])
                    
                    if 'avg_pre_volatility' in metrics and 'avg_post_volatility' in metrics and metrics['avg_pre_volatility'] > 0:
                        volatility_ratios.append(metrics['avg_post_volatility'] / metrics['avg_pre_volatility'])
            
            results['strong_form']['avg_speed_of_adjustment'] = np.mean(speed_of_adjustment) if speed_of_adjustment else None
            results['strong_form']['avg_volatility_ratio'] = np.mean(volatility_ratios) if volatility_ratios else None
    else:
        if verbose:
            print("No markets found with sufficient events for event study")
    
    # 5. Save results
    with open(os.path.join(results_dir, 'comprehensive_results.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def json_serialize(obj):
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
            return str(obj)
        
        json.dump(results, f, default=json_serialize, indent=2)
    
    # 6. Create summary visualizations
    if weak_form_results:
        # Efficiency score distribution
        plt.figure(figsize=(12, 6))
        sns.histplot([r['efficiency_score'] for r in weak_form_results], bins=10, kde=True)
        plt.axvline(x=results['aggregate']['avg_efficiency_score'], color='r', linestyle='--', 
                   label=f"Mean: {results['aggregate']['avg_efficiency_score']:.2f}")
        plt.title("Market Efficiency Score Distribution", fontsize=14)
        plt.xlabel("Efficiency Score", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(results_dir, "efficiency_score_distribution.png"), dpi=300)
        plt.close()
        
        # Efficiency by market type
        if len(weak_form_results) >= 5:
            # Group markets by type
            market_types = {}
            for result in weak_form_results:
                if 'electionType' in result:
                    market_type = result['electionType']
                    if market_type not in market_types:
                        market_types[market_type] = []
                    market_types[market_type].append(result['efficiency_score'])
            
            # Filter to types with enough data
            market_types = {k: v for k, v in market_types.items() if len(v) >= 2}
            
            if market_types:
                plt.figure(figsize=(14, 6))
                
                # Calculate average scores
                avg_scores = {k: np.mean(v) for k, v in market_types.items()}
                counts = {k: len(v) for k, v in market_types.items()}
                
                # Sort by average score
                sorted_types = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)
                
                # Create the plot
                bars = plt.bar(sorted_types, [avg_scores[t] for t in sorted_types], color='skyblue')
                
                # Add count labels
                for i, bar in enumerate(bars):
                    plt.text(i, bar.get_height() + 2, f"n={counts[sorted_types[i]]}", 
                            ha='center', va='bottom')
                
                plt.axhline(y=results['aggregate']['avg_efficiency_score'], color='r', linestyle='--',
                           label=f"Overall Average: {results['aggregate']['avg_efficiency_score']:.2f}")
                
                plt.title("Average Efficiency Score by Market Type", fontsize=14)
                plt.xlabel("Market Type", fontsize=12)
                plt.ylabel("Average Efficiency Score", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 105)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, "efficiency_by_market_type.png"), dpi=300)
                plt.close()
    
    # Time-varying efficiency visualization
    if time_varying_results:
        plt.figure(figsize=(10, 6))
        
        # Count efficiency changes
        changes = {
            'More Efficient': 0,
            'Less Efficient': 0, 
            'No Change': 0
        }
        
        for result in time_varying_results:
            if 'summary' in result:
                change = result['summary']['efficiency_change']
                changes[change] = changes.get(change, 0) + 1
        
        # Create the plot
        colors = ['green', 'red', 'gray']
        plt.bar(changes.keys(), changes.values(), color=colors)
        
        # Add percentage labels
        total = sum(changes.values())
        for i, (change, count) in enumerate(changes.items()):
            plt.text(i, count + 0.1, f"{count/total*100:.1f}%", ha='center')
        
        plt.title("Time-varying Efficiency Changes", fontsize=14)
        plt.ylabel("Number of Markets", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "time_varying_efficiency_changes.png"), dpi=300)
        plt.close()
    
    # Strong-form efficiency visualization
    if event_study_results and len(event_study_results) > 0:
        # Create average event response chart
        plt.figure(figsize=(12, 6))
        
        # Aggregate all event returns
        all_events = []
        for result in event_study_results:
            for event in result['event_results']:
                if 'cum_returns' in event:
                    all_events.append(event['cum_returns'])
        
        if all_events:
            # Resample to common time points
            common_points = np.linspace(-24, 24, 49)  # -24 to +24 hours in 1-hour steps
            resampled_events = []
            
            for event in all_events:
                if len(event) >= 5:  # Ensure enough data points
                    # Convert to numpy array for interpolation
                    event_array = np.array(list(zip(event.index, event.values)))
                    
                    # Create interpolated event
                    interpolated = np.interp(common_points, 
                                            event_array[:, 0], 
                                            event_array[:, 1],
                                            left=np.nan, right=np.nan)
                    
                    resampled_events.append(interpolated)
            
            if resampled_events:
                # Calculate average response
                avg_response = np.nanmean(resampled_events, axis=0)
                
                # Plot average response
                plt.plot(common_points, avg_response, 'b-', linewidth=2, label='Average Response')
                
                # Plot individual events (slightly transparent)
                for event in resampled_events:
                    plt.plot(common_points, event, 'gray', alpha=0.2)
                
                plt.axvline(x=0, color='r', linestyle='--', label='Event Time')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                plt.title("Average Market Response to Significant Events", fontsize=14)
                plt.xlabel("Hours Relative to Event", fontsize=12)
                plt.ylabel("Cumulative Return", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, "avg_event_response.png"), dpi=300)
                plt.close()
    
    # Generate summary report
    summary_text = []
    summary_text.append("# Polymarket Efficiency Analysis Summary Report")
    summary_text.append(f"Analyzed {results['dataset_info']['markets_analyzed']} markets\n")
    
    # Weak-form efficiency
    if weak_form_results:
        summary_text.append("## Weak-Form Efficiency")
        summary_text.append(f"Average Efficiency Score: {results['aggregate']['avg_efficiency_score']:.2f}/100")
        summary_text.append(f"Median Efficiency Score: {results['aggregate']['median_efficiency_score']:.2f}/100")
        
        # Efficiency classes
        if 'efficiency_classes' in results['aggregate']:
            summary_text.append("\nEfficiency Classification:")
            classes = results['aggregate']['efficiency_classes']
            total = sum(classes.values())
            for cls, count in classes.items():
                summary_text.append(f"- {cls}: {count} markets ({count/total*100:.1f}%)")
    
    # Time-varying efficiency
    if time_varying_results:
        summary_text.append("\n## Time-Varying Efficiency")
        
        if 'efficiency_changes' in results['time_varying']:
            changes = results['time_varying']['efficiency_changes']
            total = sum(changes.values())
            summary_text.append("Efficiency Changes Over Time:")
            for change, count in changes.items():
                summary_text.append(f"- {change}: {count} markets ({count/total*100:.1f}%)")
        
        if 'avg_volatility_change' in results['time_varying']:
            volatility_change = results['time_varying']['avg_volatility_change']
            summary_text.append(f"\nAverage Volatility Change: {volatility_change*100:.1f}%")
    
    # Strong-form efficiency
    if event_study_results:
        summary_text.append("\n## Strong-Form Efficiency")
        summary_text.append(f"Analyzed {len(event_study_results)} markets for event studies")
        
        if 'avg_speed_of_adjustment' in results['strong_form']:
            summary_text.append(f"Average Speed of Adjustment: {results['strong_form']['avg_speed_of_adjustment']:.4f}")
        
        if 'avg_volatility_ratio' in results['strong_form']:
            summary_text.append(f"Average Post-Event/Pre-Event Volatility Ratio: {results['strong_form']['avg_volatility_ratio']:.4f}")
    
    # Overall conclusion
    summary_text.append("\n## Conclusion")
    
    if weak_form_results:
        avg_score = results['aggregate']['avg_efficiency_score']
        
        if avg_score >= 75:
            summary_text.append("Polymarket election markets appear to be highly efficient, with prices following random walk patterns and little evidence of exploitable patterns.")
        elif avg_score >= 60:
            summary_text.append("Polymarket election markets appear to be moderately efficient, with some evidence of predictability but generally following random walk patterns.")
        elif avg_score >= 45:
            summary_text.append("Polymarket election markets show mixed efficiency, with significant evidence of predictable patterns in some markets.")
        else:
            summary_text.append("Polymarket election markets show signs of inefficiency, with substantial evidence of predictable patterns across many markets.")
    
    # Save summary report
    with open(os.path.join(results_dir, "efficiency_summary.md"), "w") as f:
        f.write("\n".join(summary_text))
    
    if verbose:
        print("\nAnalysis complete. Results saved to:", results_dir)
    
    return results

if __name__ == "__main__":
    run_comprehensive_analysis()