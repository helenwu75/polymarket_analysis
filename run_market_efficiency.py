# run_efficiency_analysis.py
import os
import sys
import argparse
from datetime import datetime

# Add project path
if 'src/knowledge_value' not in sys.path:
    sys.path.append('src/knowledge_value')

from market_efficiency_analysis import MarketEfficiencyAnalyzer
from strong_form_efficiency import StrongFormEfficiencyAnalyzer
from analyze_efficiency import run_comprehensive_analysis, analyze_and_report_results, generate_report

def main():
    parser = argparse.ArgumentParser(description="Run market efficiency analysis on Polymarket data")
    parser.add_argument("--data_dir", default="data", help="Directory with the Polymarket data")
    parser.add_argument("--results_dir", default="results/market_efficiency", help="Directory to save results")
    parser.add_argument("--markets", type=int, default=30, help="Maximum number of markets to analyze")
    parser.add_argument("--report", action="store_true", help="Generate a comprehensive report")
    parser.add_argument("--focus", default=None, choices=["weak", "time", "strong", "all"], 
                        help="Focus analysis on specific efficiency tests")
    
    args = parser.parse_args()
    
    # Create unique results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting market efficiency analysis at {timestamp}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Analyzing up to {args.markets} markets")
    
    # Run the analysis
    results = run_comprehensive_analysis(
        data_dir=args.data_dir,
        results_dir=results_dir,
        market_count=args.markets,
        verbose=True
    )
    
    if results:
        # Analyze and summarize results
        summary = analyze_and_report_results(args.data_dir, results_dir)
        
        if args.report and summary:
            # Generate a comprehensive report
            report_path = generate_report(summary, results_dir)
            print(f"Analysis complete. Report saved to: {report_path}")
        else:
            print(f"Analysis complete. Results saved to: {results_dir}")
    else:
        print("Analysis failed to produce results")

if __name__ == "__main__":
    main()