# strong_form_efficiency.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

class StrongFormEfficiencyAnalyzer:
    """
    Class for analyzing strong-form market efficiency through event studies
    and analysis of insider trading patterns.
    """
    
    def __init__(self, market_analyzer):
        """
        Initialize with a reference to the main market analyzer
        
        Parameters:
        -----------
        market_analyzer : MarketEfficiencyAnalyzer
            The main market analyzer instance
        """
        self.analyzer = market_analyzer
        self.results_dir = market_analyzer.results_dir
    
    def identify_significant_events(self, market_id, std_multiple=2.0):
        """
        Identify significant price movements that could be related to events
        
        Parameters:
        -----------
        market_id : str
            Market ID to analyze
        std_multiple : float
            Multiple of standard deviation to identify significant moves
            
        Returns:
        --------
        pd.Series
            Series of significant price movements indexed by date
        """
        # Get market data
        market_data = self.analyzer.preprocess_market_data(market_id)
        
        if market_data is None or len(market_data) < 30:
            print(f"Insufficient data for market {market_id}")
            return None
        
        # Calculate daily returns
        daily_data = market_data['price'].resample('D').last().dropna()
        daily_returns = daily_data.pct_change().dropna()
        
        # Identify significant moves
        threshold = std_multiple * daily_returns.std()
        significant_moves = daily_returns[abs(daily_returns) > threshold]
        
        return significant_moves
    
    def run_event_study(self, market_id, event_dates, window_size=3):
        """
        Run an event study around specified event dates
        
        Parameters:
        -----------
        market_id : str
            Market ID to analyze
        event_dates : list
            List of event dates (datetime objects)
        window_size : int
            Number of days before and after event to include in window
            
        Returns:
        --------
        dict
            Dictionary with event study results
        """
        # Get market data
        market_data = self.analyzer.preprocess_market_data(market_id)
        
        if market_data is None or len(market_data) < 30:
            print(f"Insufficient data for market {market_id}")
            return None
        
        # Ensure we have a price series
        price_series = market_data['price']
        
        # Calculate hourly returns
        hourly_data = price_series.resample('H').last().fillna(method='ffill')
        hourly_returns = hourly_data.pct_change().dropna()
        
        # Analyze each event
        event_results = []
        
        for event_date in event_dates:
            # Define event window
            start_date = event_date - timedelta(days=window_size)
            end_date = event_date + timedelta(days=window_size)
            
            # Get window data
            window_data = hourly_returns[(hourly_returns.index >= start_date) & 
                                         (hourly_returns.index <= end_date)]
            
            if len(window_data) < window_size:
                continue
            
            # Calculate cumulative returns
            cum_returns = (1 + window_data).cumprod() - 1
            
            # Normalize to event time
            cum_returns.index = (cum_returns.index - event_date).total_seconds() / 3600  # Hours from event
            
            # Calculate statistics
            pre_event = window_data[window_data.index < event_date]
            post_event = window_data[window_data.index >= event_date]
            
            event_result = {
                'event_date': event_date,
                'cum_returns': cum_returns,
                'pre_event_volatility': pre_event.std(),
                'post_event_volatility': post_event.std(),
                'pre_event_return': (1 + pre_event).prod() - 1,
                'post_event_return': (1 + post_event).prod() - 1,
                'speed_of_adjustment': None  # Will be calculated if enough data
            }
            
            # Calculate speed of adjustment if possible
            if len(post_event) > 0:
                # Speed measured by percentage of total move achieved in first hour
                total_move = event_result['post_event_return']
                first_hour = post_event.iloc[0] if len(post_event) > 0 else 0
                
                if abs(total_move) > 0:
                    event_result['speed_of_adjustment'] = first_hour / total_move
            
            event_results.append(event_result)
        
        # Calculate average metrics
        avg_metrics = {
            'avg_pre_volatility': np.mean([r['pre_event_volatility'] for r in event_results]),
            'avg_post_volatility': np.mean([r['post_event_volatility'] for r in event_results]),
            'avg_speed_of_adjustment': np.mean([r['speed_of_adjustment'] for r in event_results 
                                               if r['speed_of_adjustment'] is not None])
        }
        
        return {
            'market_id': market_id,
            'event_results': event_results,
            'avg_metrics': avg_metrics
        }
    
    def visualize_event_study(self, event_study_results):
        """
        Create visualization for event study results
        
        Parameters:
        -----------
        event_study_results : dict
            Results from run_event_study method
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with event study visualization
        """
        if not event_study_results or 'event_results' not in event_study_results:
            print("No event study results to visualize")
            return None
        
        event_results = event_study_results['event_results']
        
        if not event_results:
            print("No events found in results")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot cumulative returns around events
        for i, event in enumerate(event_results):
            cum_returns = event['cum_returns']
            ax1.plot(cum_returns.index, cum_returns.values, 
                     label=f"Event {i+1}: {event['event_date'].date()}")
        
        ax1.axvline(x=0, color='r', linestyle='--', label='Event Time')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_title("Cumulative Returns Around Events", fontsize=14)
        ax1.set_xlabel("Hours Relative to Event", fontsize=12)
        ax1.set_ylabel("Cumulative Return", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot volatility comparison
        pre_vols = [r['pre_event_volatility'] for r in event_results]
        post_vols = [r['post_event_volatility'] for r in event_results]
        
        bar_positions = np.arange(len(event_results))
        bar_width = 0.35
        
        ax2.bar(bar_positions - bar_width/2, pre_vols, bar_width, label='Pre-Event')
        ax2.bar(bar_positions + bar_width/2, post_vols, bar_width, label='Post-Event')
        
        # Add event labels
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels([f"Event {i+1}" for i in range(len(event_results))])
        
        ax2.set_title("Volatility Comparison", fontsize=14)
        ax2.set_ylabel("Return Volatility", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add summary metrics
        avg_metrics = event_study_results['avg_metrics']
        plt.figtext(0.5, 0.01, 
                   f"Avg Speed of Adjustment: {avg_metrics['avg_speed_of_adjustment']:.2f} | " +
                   f"Avg Pre-Event Vol: {avg_metrics['avg_pre_volatility']:.4f} | " +
                   f"Avg Post-Event Vol: {avg_metrics['avg_post_volatility']:.4f}",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig