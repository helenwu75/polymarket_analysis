#!/usr/bin/env python3
'''
This script runs the market efficiency analysis on the Polymarket dataset.
'''

from src.knowledge_value.market_efficiency import MarketEfficiencyAnalyzer

def main():
    # Initialize the analyzer
    analyzer = MarketEfficiencyAnalyzer(
        data_dir='data',
        results_dir='results/knowledge_value/efficiency'
    )
    
    # Run the complete analysis
    analyzer.run_analysis(
        max_markets=10,  # Analyze the top 100 markets by volume
        max_events=2     # Analyze the top 20 events with multiple markets
    )

if __name__ == "__main__":
    main()