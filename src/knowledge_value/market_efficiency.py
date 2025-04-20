#!/usr/bin/env python3
'''
Market Efficiency Analysis

This module implements tests for market efficiency in prediction markets based on:
1. Weak-form efficiency tests (market-level)
2. Cross-market predictability (event-level)
3. Time-varying efficiency analysis
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import os
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv('PYTHONPATH'))



# Import existing utility functions
from src.utils.data_loader import load_main_dataset, load_trade_data

class MarketEfficiencyAnalyzer:
    """Analyzes market efficiency for prediction markets."""
    
    def __init__(self, data_dir='data', results_dir='results/knowledge_value/efficiency'):
        """
        Initialize the analyzer with data directory paths.
        
        Parameters:
        -----------
        data_dir : str
            Path to the data directory containing cleaned_election_data.csv and trades/
        results_dir : str
            Path to save results and plots
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load main dataset
        self.main_df = load_main_dataset(f"{data_dir}/cleaned_election_data.csv")
        print(f"Loaded main dataset with {self.main_df.shape[0]} rows and {self.main_df.shape[1]} columns")
        
        # Initialize results storage
        self.results = {
            'weak_form': {},
            'cross_market': {},
            'time_varying': {}
        }
    
    def preprocess_market_data(self, market_id, resample='1min'):
        """
        Convert raw trade data to time series of prices and returns.
        
        Parameters:
        -----------
        market_id : str
            The ID of the market to analyze
        resample : str
            Frequency to resample the time series (default: '1min')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: timestamp, price, log_return
        """
        # Load trade data for the specific market
        trades_df = load_trade_data(market_id, trades_dir=f"{self.data_dir}/trades")
        
        if trades_df is None or len(trades_df) < 30:
            print(f"Insufficient trade data for market {market_id}")
            return None
        
        # Ensure timestamp is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Sort by timestamp
        trades_df = trades_df.sort_values('timestamp')
        
        # Ensure price is numeric
        if 'price' in trades_df.columns:
            trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
        elif 'price_num' in trades_df.columns:
            trades_df['price'] = pd.to_numeric(trades_df['price_num'], errors='coerce')
        else:
            print(f"No price column found for market {market_id}")
            return None
        
        # Drop rows with NaN prices
        trades_df = trades_df.dropna(subset=['price'])
        
        # Resample to regular intervals
        trades_df = trades_df.set_index('timestamp')
        price_series = trades_df['price'].resample(resample).last()
        
        # Fill missing values using forward fill (using ffill instead of method='ffill')
        price_series = price_series.ffill()
        
        # Calculate log returns
        log_returns = np.log(price_series / price_series.shift(1))
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'price': price_series,
            'log_return': log_returns
        })
        
        # Drop rows with NaN
        result_df = result_df.dropna()
        
        return result_df
        
    def run_autocorrelation_tests(self, returns, market_id, lags=[60, 360, 1440]):
        """
        Run ACF/PACF tests on return series.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        market_id : str
            ID of the market for labeling results
        lags : list
            List of lag periods to test (in minutes)
            
        Returns:
        --------
        dict
            Dictionary with ACF/PACF results and significance
        """
        results = {}
        
        for lag in lags:
            # Limit lag to length of series
            effective_lag = min(lag, len(returns) - 1)
            
            if effective_lag < 5:  # Skip if too few observations
                continue
                
            # Calculate ACF and PACF
            acf_values = acf(returns, nlags=effective_lag, fft=True)
            pacf_values = pacf(returns, nlags=effective_lag)
            
            # Test statistical significance using Ljung-Box test
            lb_test = acorr_ljungbox(returns, lags=[effective_lag])
            lb_stat = lb_test.iloc[0, 0]
            lb_pvalue = lb_test.iloc[0, 1]
            
            # Create result entry
            lag_key = f"{effective_lag}min"
            results[lag_key] = {
                'acf': acf_values.tolist(),
                'pacf': pacf_values.tolist(),
                'ljung_box_stat': lb_stat,
                'ljung_box_pvalue': lb_pvalue,
                'significant': lb_pvalue < 0.05
            }
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            plot_acf(returns, lags=effective_lag, ax=ax1, title=f'ACF - Market {market_id} - Lag {lag_key}')
            plot_pacf(returns, lags=effective_lag, ax=ax2, title=f'PACF - Market {market_id} - Lag {lag_key}')
            plt.tight_layout()
            
            # Save plots
            plt.savefig(f"{self.results_dir}/acf_pacf_{market_id}_{lag_key}.png")
            plt.close()
        
        return results
    
    
    def run_adf_test(self, series, series_type='price'):
        """
        Run Augmented Dickey-Fuller test for unit root.
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        series_type : str
            Type of series ('price' or 'return')
            
        Returns:
        --------
        dict
            Dictionary with test results
        """
        # Run ADF test
        result = adfuller(series.dropna())
        
        # Format results
        adf_result = {
            'adf_statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05  # Reject unit root if p-value < 0.05
        }
        
        return adf_result
    
    def run_variance_ratio_test(self, returns, market_id, periods=[1, 5, 15, 60]):
        """
        Run variance ratio test to check if variance scales linearly with time.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        market_id : str
            ID of the market
        periods : list
            List of periods to test
            
        Returns:
        --------
        dict
            Dictionary with variance ratio results
        """
        results = {}
        
        # Calculate variance for base period (typically 1-minute)
        base_period = periods[0]
        base_var = returns.var()
        
        for period in periods[1:]:
            # Skip if we don't have enough data
            if len(returns) < period * 10:
                continue
                
            # Aggregate returns for longer period
            agg_returns = returns.rolling(window=period).sum()
            agg_returns = agg_returns[~np.isnan(agg_returns)]
            
            if len(agg_returns) <= 1:
                continue
                
            # Calculate variance
            period_var = agg_returns.var()
            
            # Calculate variance ratio
            var_ratio = period_var / (period * base_var)
            
            # Random walk hypothesis: var_ratio should be close to 1
            # Calculate z-statistic (simplified)
            n = len(returns)
            std_error = np.sqrt(2 * (2 * period - 1) * (period - 1) / (3 * period * n))
            z_stat = (var_ratio - 1) / std_error
            p_value = 2 * (1 - abs(np.exp(-0.5 * z_stat**2) / np.sqrt(2 * np.pi)))
            
            results[f"{period}min"] = {
                'variance_ratio': var_ratio,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Mean Reversion' if var_ratio < 1 else 'Momentum' if var_ratio > 1 else 'Random Walk'
            }
        
        return results
    
    def fit_ar_model(self, returns, market_id, order=1, train_ratio=0.8):
        """
        Fit AR model to return series and evaluate predictability.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        market_id : str
            ID of the market
        order : int
            Order of the AR model
        train_ratio : float
            Ratio of data to use for training
            
        Returns:
        --------
        dict
            Dictionary with model results
        """
        if len(returns) <= order + 2:
            return None
            
        # Split into train and test
        train_size = int(len(returns) * train_ratio)
        train, test = returns[:train_size], returns[train_size:]
        
        if len(train) <= order + 1 or len(test) == 0:
            return None
            
        # Fit AR model
        try:
            model = AutoReg(train, lags=order)
            model_fit = model.fit()
            
            # Make predictions
            predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
            
            # Calculate metrics
            mse = ((predictions - test) ** 2).mean()
            mae = abs(predictions - test).mean()
            
            # Calculate R² for out-of-sample predictions
            ss_tot = ((test - test.mean()) ** 2).sum()
            ss_res = ((test - predictions) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Extract coefficient and p-value
            coef = model_fit.params[1] if len(model_fit.params) > 1 else 0
            p_value = model_fit.pvalues[1] if len(model_fit.pvalues) > 1 else 1
            
            return {
                'ar_coefficient': coef,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'out_of_sample_r2': r2,
                'mse': mse,
                'mae': mae,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        except Exception as e:
            print(f"Error fitting AR model for market {market_id}: {e}")
            return None
    
    def analyze_weak_form_efficiency(self, market_ids=None, max_markets=100):
        """
        Analyze weak-form efficiency for selected markets with expanded tests.
        
        Parameters:
        -----------
        market_ids : list
            List of market IDs to analyze. If None, a sample will be selected.
        max_markets : int
            Maximum number of markets to analyze if market_ids is None
            
        Returns:
        --------
        dict
            Dictionary with weak-form efficiency results
        """
        if market_ids is None:
            # Determine ID column
            id_column = None
            if 'market_id' in self.main_df.columns:
                id_column = 'market_id'
            elif 'id' in self.main_df.columns:
                id_column = 'id'
            else:
                id_column = self.main_df.columns[0]
                print(f"Warning: Using {id_column} as market ID column")
            
            # Sort by volume if available
            sort_column = 'volumeNum' if 'volumeNum' in self.main_df.columns else id_column
            markets_df = self.main_df.sort_values(sort_column, ascending=False)
            market_ids = markets_df[id_column].unique()[:max_markets]
        
        results = {}
        
        for market_id in tqdm(market_ids, desc="Analyzing weak-form efficiency"):
            try:
                market_result = {'market_id': market_id}
                
                # Preprocess market data
                market_data = self.preprocess_market_data(market_id)
                if market_data is None or len(market_data) < 30:
                    print(f"Skipping market {market_id}: insufficient data")
                    continue
                    
                # Store market metadata
                # Get market info using ID column
                id_column = None
                if 'market_id' in self.main_df.columns:
                    id_column = 'market_id'
                elif 'id' in self.main_df.columns:
                    id_column = 'id'
                else:
                    id_column = self.main_df.columns[0]
                
                # Find the row for this market safely
                market_rows = self.main_df[self.main_df[id_column] == market_id]
                if len(market_rows) == 0:
                    # Try string comparison if numerical comparison fails
                    market_rows = self.main_df[self.main_df[id_column].astype(str) == str(market_id)]
                
                if len(market_rows) > 0:
                    market_info = market_rows.iloc[0]
                    
                    # Get values safely using dictionary access or attribute access with fallback
                    try:
                        if hasattr(market_info, 'get'):  # If it's dictionary-like
                            market_result['event_type'] = market_info.get('event_electionType', 'Unknown')
                            market_result['country'] = market_info.get('event_country', 'Unknown')
                            market_result['volume'] = market_info.get('volumeNum', 0)
                            market_result['duration_days'] = market_info.get('market_duration_days', 0)
                        else:  # If it's Series-like
                            market_result['event_type'] = market_info['event_electionType'] if 'event_electionType' in market_info else 'Unknown'
                            market_result['country'] = market_info['event_country'] if 'event_country' in market_info else 'Unknown'
                            market_result['volume'] = market_info['volumeNum'] if 'volumeNum' in market_info else 0
                            market_result['duration_days'] = market_info['market_duration_days'] if 'market_duration_days' in market_info else 0
                    except (KeyError, TypeError) as e:
                        print(f"Error retrieving market metadata for {market_id}: {e}")
                        # Set defaults
                        market_result['event_type'] = 'Unknown'
                        market_result['country'] = 'Unknown'
                        market_result['volume'] = 0
                        market_result['duration_days'] = 0
                
                # Run ADF tests
                market_result['adf_price'] = self.run_adf_test(market_data['price'], 'price')
                market_result['adf_return'] = self.run_adf_test(market_data['log_return'], 'return')
                
                # Run autocorrelation tests
                market_result['autocorrelation'] = self.run_autocorrelation_tests(
                    market_data['log_return'], market_id
                )
                
                # Run variance ratio test
                market_result['variance_ratio'] = self.run_variance_ratio_test(
                    market_data['log_return'], market_id
                )
                
                # Run runs test
                market_result['runs_test'] = self.run_runs_test(market_data['log_return'])
                
                # Run time-varying efficiency analysis - pass both returns and market_id
                market_result['time_varying'] = self.analyze_time_varying_efficiency(
                    market_data['log_return'], market_id
                )
                
                # Fit AR model
                market_result['ar_model'] = self.fit_ar_model(
                    market_data['log_return'], market_id
                )
                
                # Store results
                results[market_id] = market_result
                
            except Exception as e:
                print(f"Error analyzing market {market_id}: {e}")
                continue
        
        # Summarize results
        if results:
            self._summarize_weak_form_results(results)
        
        return results

    def _summarize_weak_form_results(self, results):
        """Print a summary of weak-form efficiency results"""
        if not results:
            print("No results to summarize")
            return
        
        total_markets = len(results)
        
        # Count key metrics
        ar_significant = sum(1 for r in results.values() if r.get('ar_model', {}).get('significant', False))
        adf_price_stationary = sum(1 for r in results.values() if r.get('adf_price', {}).get('is_stationary', False))
        adf_return_stationary = sum(1 for r in results.values() if r.get('adf_return', {}).get('is_stationary', False))
        runs_random = sum(1 for r in results.values() if r.get('runs_test', {}).get('is_random', True))
        
        # Count variance ratio results
        vr_mean_reversion = 0
        vr_momentum = 0
        vr_random_walk = 0
        
        for r in results.values():
            vr = r.get('variance_ratio', {})
            for period_result in vr.values():
                if period_result.get('interpretation') == 'Mean Reversion':
                    vr_mean_reversion += 1
                    break  # Count market once
            for period_result in vr.values():
                if period_result.get('interpretation') == 'Momentum':
                    vr_momentum += 1
                    break  # Count market once
            for period_result in vr.values():
                if period_result.get('interpretation') == 'Random Walk':
                    vr_random_walk += 1
                    break  # Count market once
        
        # Print summary
        print("\n" + "="*50)
        print(f"WEAK-FORM EFFICIENCY RESULTS SUMMARY ({total_markets} markets)")
        print("="*50)
        
        print(f"\nPrice stationarity: {adf_price_stationary} markets ({adf_price_stationary/total_markets*100:.1f}%)")
        print(f"Return stationarity: {adf_return_stationary} markets ({adf_return_stationary/total_markets*100:.1f}%)")
        print(f"Significant AR(1): {ar_significant} markets ({ar_significant/total_markets*100:.1f}%)")
        print(f"Random by runs test: {runs_random} markets ({runs_random/total_markets*100:.1f}%)")
        
        print("\nVariance Ratio Results:")
        print(f"  Mean Reversion: {vr_mean_reversion} markets ({vr_mean_reversion/total_markets*100:.1f}%)")
        print(f"  Momentum: {vr_momentum} markets ({vr_momentum/total_markets*100:.1f}%)")
        print(f"  Random Walk: {vr_random_walk} markets ({vr_random_walk/total_markets*100:.1f}%)")
        
        # Group by market type if available
        event_types = set()
        for r in results.values():
            if 'event_type' in r and r['event_type'] != 'Unknown':
                event_types.add(r['event_type'])
        
        if event_types:
            print("\nEfficiency by Market Type:")
            for event_type in sorted(event_types):
                type_markets = [r for r in results.values() if r.get('event_type') == event_type]
                type_count = len(type_markets)
                if type_count < 5:  # Skip if too few markets
                    continue
                    
                type_ar_significant = sum(1 for r in type_markets if r.get('ar_model', {}).get('significant', False))
                
                print(f"  {event_type} ({type_count} markets): {type_ar_significant/type_count*100:.1f}% show inefficiency")
    
    def _aggregate_weak_form_results(self, market_results):
        """Aggregate weak-form efficiency results across markets."""
        if not market_results:
            return {}
            
        aggregate = {
            'total_markets': len(market_results),
            'autocorrelation': {
                'significant_markets': 0,
                'significant_percentage': 0,
            },
            'adf_price': {
                'stationary_markets': 0,
                'stationary_percentage': 0,
            },
            'adf_return': {
                'stationary_markets': 0,
                'stationary_percentage': 0,
            },
            'variance_ratio': {
                'significant_markets': 0,
                'significant_percentage': 0,
                'mean_reversion_markets': 0,
                'momentum_markets': 0,
            },
            'ar_model': {
                'significant_markets': 0,
                'significant_percentage': 0,
                'average_r2': 0,
                'positive_coefficient_markets': 0,
                'negative_coefficient_markets': 0,
            },
            'by_election_type': {},
            'by_country': {},
        }
        
        # Count features
        valid_markets = 0
        for market_id, results in market_results.items():
            valid_markets += 1
            
            # Count autocorrelation
            if 'autocorrelation' in results:
                any_significant = any(lag_result.get('significant', False) 
                                     for lag_result in results['autocorrelation'].values())
                if any_significant:
                    aggregate['autocorrelation']['significant_markets'] += 1
            
            # Count ADF tests
            if 'adf_price' in results and results['adf_price'].get('is_stationary', False):
                aggregate['adf_price']['stationary_markets'] += 1
                
            if 'adf_return' in results and results['adf_return'].get('is_stationary', False):
                aggregate['adf_return']['stationary_markets'] += 1
            
            # Count variance ratio tests
            if 'variance_ratio' in results:
                any_significant = any(vr_result.get('significant', False) 
                                     for vr_result in results['variance_ratio'].values())
                if any_significant:
                    aggregate['variance_ratio']['significant_markets'] += 1
                
                for vr_result in results['variance_ratio'].values():
                    if vr_result.get('interpretation') == 'Mean Reversion':
                        aggregate['variance_ratio']['mean_reversion_markets'] += 1
                        break
                        
                for vr_result in results['variance_ratio'].values():
                    if vr_result.get('interpretation') == 'Momentum':
                        aggregate['variance_ratio']['momentum_markets'] += 1
                        break
            
            # Count AR model results
            if 'ar_model' in results and results['ar_model'] is not None:
                if results['ar_model'].get('significant', False):
                    aggregate['ar_model']['significant_markets'] += 1
                
                if results['ar_model'].get('ar_coefficient', 0) > 0:
                    aggregate['ar_model']['positive_coefficient_markets'] += 1
                elif results['ar_model'].get('ar_coefficient', 0) < 0:
                    aggregate['ar_model']['negative_coefficient_markets'] += 1
                    
                aggregate['ar_model']['average_r2'] += results['ar_model'].get('out_of_sample_r2', 0)
            
            # Group by election type
            election_type = results.get('event_type', 'Unknown')
            if election_type not in aggregate['by_election_type']:
                aggregate['by_election_type'][election_type] = {
                    'count': 0,
                    'significant_autocorrelation': 0,
                    'stationary_price': 0,
                    'significant_ar': 0,
                }
            
            aggregate['by_election_type'][election_type]['count'] += 1
            
            if 'autocorrelation' in results and any(lag_result.get('significant', False) 
                                                 for lag_result in results['autocorrelation'].values()):
                aggregate['by_election_type'][election_type]['significant_autocorrelation'] += 1
                
            if 'adf_price' in results and results['adf_price'].get('is_stationary', False):
                aggregate['by_election_type'][election_type]['stationary_price'] += 1
                
            if 'ar_model' in results and results['ar_model'] is not None and results['ar_model'].get('significant', False):
                aggregate['by_election_type'][election_type]['significant_ar'] += 1
            
            # Group by country
            country = results.get('country', 'Unknown')
            if country not in aggregate['by_country']:
                aggregate['by_country'][country] = {
                    'count': 0,
                    'significant_autocorrelation': 0,
                    'stationary_price': 0,
                    'significant_ar': 0,
                }
            
            aggregate['by_country'][country]['count'] += 1
            
            if 'autocorrelation' in results and any(lag_result.get('significant', False) 
                                                 for lag_result in results['autocorrelation'].values()):
                aggregate['by_country'][country]['significant_autocorrelation'] += 1
                
            if 'adf_price' in results and results['adf_price'].get('is_stationary', False):
                aggregate['by_country'][country]['stationary_price'] += 1
                
            if 'ar_model' in results and results['ar_model'] is not None and results['ar_model'].get('significant', False):
                aggregate['by_country'][country]['significant_ar'] += 1
        
        # Calculate percentages
        if valid_markets > 0:
            aggregate['autocorrelation']['significant_percentage'] = aggregate['autocorrelation']['significant_markets'] / valid_markets * 100
            aggregate['adf_price']['stationary_percentage'] = aggregate['adf_price']['stationary_markets'] / valid_markets * 100
            aggregate['adf_return']['stationary_percentage'] = aggregate['adf_return']['stationary_markets'] / valid_markets * 100
            aggregate['variance_ratio']['significant_percentage'] = aggregate['variance_ratio']['significant_markets'] / valid_markets * 100
            aggregate['ar_model']['significant_percentage'] = aggregate['ar_model']['significant_markets'] / valid_markets * 100
            
            if aggregate['ar_model']['significant_markets'] > 0:
                aggregate['ar_model']['average_r2'] /= aggregate['ar_model']['significant_markets']
            
            # Calculate percentages for election types
            for election_type in aggregate['by_election_type']:
                count = aggregate['by_election_type'][election_type]['count']
                if count > 0:
                    aggregate['by_election_type'][election_type]['significant_autocorrelation_pct'] = (
                        aggregate['by_election_type'][election_type]['significant_autocorrelation'] / count * 100
                    )
                    aggregate['by_election_type'][election_type]['stationary_price_pct'] = (
                        aggregate['by_election_type'][election_type]['stationary_price'] / count * 100
                    )
                    aggregate['by_election_type'][election_type]['significant_ar_pct'] = (
                        aggregate['by_election_type'][election_type]['significant_ar'] / count * 100
                    )
            
            # Calculate percentages for countries
            for country in aggregate['by_country']:
                count = aggregate['by_country'][country]['count']
                if count > 0:
                    aggregate['by_country'][country]['significant_autocorrelation_pct'] = (
                        aggregate['by_country'][country]['significant_autocorrelation'] / count * 100
                    )
                    aggregate['by_country'][country]['stationary_price_pct'] = (
                        aggregate['by_country'][country]['stationary_price'] / count * 100
                    )
                    aggregate['by_country'][country]['significant_ar_pct'] = (
                        aggregate['by_country'][country]['significant_ar'] / count * 100
                    )
        
        return aggregate
    
    def run_variance_ratio_test(self, returns, market_id, periods=[1, 5, 15, 60]):
        """
        Run variance ratio test to check if variance scales linearly with time.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        market_id : str
            ID of the market for labeling results
        periods : list
            List of periods to test
            
        Returns:
        --------
        dict
            Dictionary with variance ratio results
        """
        results = {}
        
        # Calculate variance for base period (typically 1-minute)
        base_period = periods[0]
        base_var = returns.var()
        
        for period in periods[1:]:
            # Skip if we don't have enough data
            if len(returns) < period * 10:
                continue
                
            # Aggregate returns for longer period
            agg_returns = returns.rolling(window=period).sum()
            agg_returns = agg_returns[~np.isnan(agg_returns)]
            
            if len(agg_returns) <= 1:
                continue
                
            # Calculate variance
            period_var = agg_returns.var()
            
            # Calculate variance ratio
            var_ratio = period_var / (period * base_var)
            
            # Random walk hypothesis: var_ratio should be close to 1
            # Calculate z-statistic (simplified)
            n = len(returns)
            std_error = np.sqrt(2 * (2 * period - 1) * (period - 1) / (3 * period * n))
            z_stat = (var_ratio - 1) / std_error
            p_value = 2 * (1 - abs(np.exp(-0.5 * z_stat**2) / np.sqrt(2 * np.pi)))
            
            results[f"{period}min"] = {
                'variance_ratio': var_ratio,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Mean Reversion' if var_ratio < 1 else 'Momentum' if var_ratio > 1 else 'Random Walk'
            }
        
            return results
        
    def run_runs_test(self, returns):
        """
        Run a runs test to check for non-random patterns in returns.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
            
        Returns:
        --------
        dict
            Dictionary with runs test results
        """
        # Convert returns to binary sequence (1 for positive, 0 for negative)
        binary_seq = (returns > 0).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq.iloc[i] != binary_seq.iloc[i-1]:  # Use iloc for positional indexing
                runs += 1
        
        # Calculate expected runs and variance
        n = len(binary_seq)
        n1 = binary_seq.sum()  # Count of 1s
        n0 = n - n1  # Count of 0s
        
        if n0 == 0 or n1 == 0:  # All returns are positive or negative
            return {
                'runs': runs,
                'expected_runs': np.nan,
                'z_statistic': np.nan,
                'p_value': np.nan,
                'is_random': False
            }
        
        expected_runs = 1 + 2 * n1 * n0 / n
        std_runs = np.sqrt(2 * n1 * n0 * (2 * n1 * n0 - n) / (n**2 * (n-1)))
        
        # Calculate z-statistic
        z_stat = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - abs(np.exp(-0.5 * z_stat**2) / np.sqrt(2 * np.pi)))
        
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_random': p_value >= 0.05  # Null hypothesis is randomness
        }

    def analyze_time_varying_efficiency(self, returns, market_id=None):
        """
        Analyze how efficiency changes over time by dividing the returns series 
        into early, middle, and late periods.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        market_id : str, optional
            ID of the market for additional metadata (if available)
            
        Returns:
        --------
        dict
            Dictionary with time-varying efficiency results
        """
        if len(returns) < 90:  # Need enough data to divide
            return None
        
        # Divide into three periods
        period_size = len(returns) // 3
        early_returns = returns.iloc[:period_size]
        mid_returns = returns.iloc[period_size:2*period_size]
        late_returns = returns.iloc[2*period_size:]
        
        # Test each period
        periods = {
            'early': early_returns,
            'middle': mid_returns,
            'late': late_returns
        }
        
        results = {}
        
        for period_name, period_returns in periods.items():
            if len(period_returns) < 30:  # Skip if not enough data
                continue
            
            # Run efficiency tests
            acf_values = acf(period_returns, nlags=5, fft=True)
            significant_acf = any(abs(acf_values[1:]) > 1.96 / np.sqrt(len(period_returns)))
            
            # AR(1) test
            ar1_result = None
            try:
                model = AutoReg(period_returns, lags=1)
                model_fit = model.fit()
                ar1_coef = model_fit.params[1] if len(model_fit.params) > 1 else 0
                ar1_pvalue = model_fit.pvalues[1] if len(model_fit.pvalues) > 1 else 1
                ar1_result = {
                    'coefficient': ar1_coef,
                    'p_value': ar1_pvalue,
                    'significant': ar1_pvalue < 0.05
                }
            except Exception as e:
                print(f"Error fitting AR model for period {period_name}: {e}")
            
            results[period_name] = {
                'significant_acf': significant_acf,
                'ar1': ar1_result,
                'return_volatility': period_returns.std(),
                'sample_size': len(period_returns)
            }
        
        # Compare early vs late
        if 'early' in results and 'late' in results:
            results['efficiency_change'] = {
                'acf_change': results['early']['significant_acf'] != results['late']['significant_acf'],
                'volatility_change': (results['late']['return_volatility'] / results['early']['return_volatility']) - 1 if results['early']['return_volatility'] > 0 else 0
            }
            
            if results['early'].get('ar1') and results['late'].get('ar1'):
                results['efficiency_change']['ar1_significance_change'] = results['early']['ar1']['significant'] != results['late']['ar1']['significant']
                results['efficiency_change']['ar1_coefficient_change'] = results['late']['ar1']['coefficient'] - results['early']['ar1']['coefficient']
        
        return results

    def analyze_granger_causality(self, event_id, lags=1):
        """
        Test Granger causality between markets in the same event.
        
        Parameters:
        -----------
        event_id : str
            ID of the event to analyze
        lags : int
            Number of lags to include in the test
            
        Returns:
        --------
        dict
            Dictionary with Granger causality results
        """
        # Get all markets in the event
        event_markets = self.main_df[self.main_df['event_id'] == event_id]
        market_ids = event_markets['market_id'].unique()
        
        if len(market_ids) <= 1:
            return None
        
        results = {
            'event_id': event_id,
            'market_pairs': [],
            'significant_pairs': 0,
            'total_pairs': 0
        }
        
        # Process each market to get time series
        market_data = {}
        for market_id in market_ids:
            data = self.preprocess_market_data(market_id, resample='5min')  # Use 5-min intervals for cross-market analysis
            if data is not None and len(data) > 10:  # Minimum length for testing
                market_data[market_id] = data
        
        # Test Granger causality for each pair of markets
        for i, market_i in enumerate(market_data.keys()):
            for j, market_j in enumerate(market_data.keys()):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                
                results['total_pairs'] += 2  # Test both directions
                
                # Align time series
                common_index = market_data[market_i].index.intersection(market_data[market_j].index)
                if len(common_index) <= lags + 2:  # Not enough data
                    continue
                
                series_i = market_data[market_i].loc[common_index, 'price']
                series_j = market_data[market_j].loc[common_index, 'price']
                
                # Test if market i Granger-causes market j
                try:
                    gc_result_ij = grangercausalitytests(
                        pd.concat([series_j, series_i], axis=1), 
                        maxlag=lags, 
                        verbose=False
                    )
                    p_value_ij = gc_result_ij[0][lags][0]['ssr_chi2test'][1]
                    significant_ij = p_value_ij < 0.05
                    
                    if significant_ij:
                        results['significant_pairs'] += 1
                    
                    # Test if market j Granger-causes market i
                    gc_result_ji = grangercausalitytests(
                        pd.concat([series_i, series_j], axis=1), 
                        maxlag=lags, 
                        verbose=False
                    )
                    p_value_ji = gc_result_ji[0][lags][0]['ssr_chi2test'][1]
                    significant_ji = p_value_ji < 0.05
                    
                    if significant_ji:
                        results['significant_pairs'] += 1
                    
                    # Store results
                    results['market_pairs'].append({
                        'market_i': market_i,
                        'market_j': market_j,
                        'i_causes_j_pvalue': p_value_ij,
                        'j_causes_i_pvalue': p_value_ji,
                        'i_causes_j': significant_ij,
                        'j_causes_i': significant_ji,
                        'bidirectional': significant_ij and significant_ji
                    })
                    
                except Exception as e:
                    print(f"Error testing Granger causality for markets {market_i} and {market_j}: {e}")
        
        return results
    
    def analyze_cross_market_predictability(self, max_events=20):
        """
        Analyze cross-market predictability for events.
        
        Parameters:
        -----------
        max_events : int
            Maximum number of events to analyze
            
        Returns:
        --------
        dict
            Dictionary with cross-market predictability results
        """
        # Get events with multiple markets
        event_counts = self.main_df.groupby('event_id').size()
        events_with_multiple_markets = event_counts[event_counts > 1].index.tolist()
        
        if max_events < len(events_with_multiple_markets):
            # Select events with the most markets
            event_counts = event_counts[events_with_multiple_markets]
            events_with_multiple_markets = event_counts.sort_values(ascending=False).index[:max_events].tolist()
        
        results = {}
        
        for event_id in tqdm(events_with_multiple_markets, desc="Analyzing cross-market predictability"):
            granger_results = self.analyze_granger_causality(event_id)
            if granger_results:
                results[event_id] = granger_results
        
        # Aggregate results
        self.results['cross_market'] = self._aggregate_cross_market_results(results)
        
        return results
    
    def _aggregate_cross_market_results(self, event_results):
        """Aggregate cross-market predictability results."""
        if not event_results:
            return {}
            
        aggregate = {
            'total_events': len(event_results),
            'total_pairs_tested': sum(event['total_pairs'] for event in event_results.values() if 'total_pairs' in event),
            'significant_pairs': sum(event['significant_pairs'] for event in event_results.values() if 'significant_pairs' in event),
            'bidirectional_pairs': 0,
            'events_with_predictability': 0,
        }
        
        # Count bidirectional pairs and events with predictability
        for event_id, event in event_results.items():
            has_predictability = event['significant_pairs'] > 0
            if has_predictability:
                aggregate['events_with_predictability'] += 1
            
            for pair in event.get('market_pairs', []):
                if pair.get('bidirectional', False):
                    aggregate['bidirectional_pairs'] += 1
        
        # Calculate percentages
        if aggregate['total_pairs_tested'] > 0:
            aggregate['significant_percentage'] = aggregate['significant_pairs'] / aggregate['total_pairs_tested'] * 100
            aggregate['bidirectional_percentage'] = aggregate['bidirectional_pairs'] / aggregate['total_pairs_tested'] * 100
        
        if aggregate['total_events'] > 0:
            aggregate['events_with_predictability_percentage'] = aggregate['events_with_predictability'] / aggregate['total_events'] * 100
        
        return aggregate
    
    def analyze_time_varying_efficiency(self, market_ids=None, max_markets=50):
        """
        Analyze how efficiency changes over time for markets.
        
        Parameters:
        -----------
        market_ids : list
            List of market IDs to analyze. If None, a sample will be selected.
        max_markets : int
            Maximum number of markets to analyze if market_ids is None
            
        Returns:
        --------
        dict
            Dictionary with time-varying efficiency results
        """
        if market_ids is None:
            # If no specific markets are provided, select a sample based on duration and volume
            markets_df = self.main_df[self.main_df['market_duration_days'] > 7]  # At least a week long
            markets_df = markets_df.sort_values('volumeNum', ascending=False)
            market_ids = markets_df['market_id'].unique()[:max_markets]
        
        results = {}
        
        for market_id in tqdm(market_ids, desc="Analyzing time-varying efficiency"):
            market_result = {'market_id': market_id}
            
            # Preprocess market data
            market_data = self.preprocess_market_data(market_id)
            if market_data is None or len(market_data) < 90:  # Need enough data to split
                continue
                
            # Store market metadata
            market_info = self.main_df[self.main_df['market_id'] == market_id].iloc[0]
            market_result['event_type'] = market_info.get('event_electionType', 'Unknown')
            market_result['country'] = market_info.get('event_country', 'Unknown')
            market_result['volume'] = market_info.get('volumeNum', 0)
            market_result['duration_days'] = market_info.get('market_duration_days', 0)
            
            # Split the data into three time periods
            total_rows = len(market_data)
            period_size = total_rows // 3
            
            early_period = market_data.iloc[:period_size]
            middle_period = market_data.iloc[period_size:2*period_size]
            late_period = market_data.iloc[2*period_size:]
            
            periods = {
                'early': early_period,
                'middle': middle_period,
                'late': late_period
            }
            
            # Run tests for each period
            period_results = {}
            
            for period_name, period_data in periods.items():
                if len(period_data) < 30:  # Skip if not enough data
                    continue
                
                period_result = {}
                
                # Run autocorrelation test
                period_result['autocorrelation'] = self.run_autocorrelation_tests(
                    period_data['log_return'], f"{market_id}_{period_name}"
                )
                
                # Run ADF test
                period_result['adf_price'] = self.run_adf_test(
                    period_data['price'], 'price'
                )
                
                period_result['adf_return'] = self.run_adf_test(
                    period_data['log_return'], 'return'
                )
                
                # Fit AR model
                period_result['ar_model'] = self.fit_ar_model(
                    period_data['log_return'], f"{market_id}_{period_name}"
                )
                
                period_results[period_name] = period_result
            
            market_result['periods'] = period_results
            
            # Compare efficiency across periods
            market_result['efficiency_evolution'] = self._compare_periods(period_results)
            
            # Store results
            results[market_id] = market_result
        
        # Aggregate results
        self.results['time_varying'] = self._aggregate_time_varying_results(results)
        
        return results
    
    def _compare_periods(self, period_results):
        """Compare efficiency metrics across time periods."""
        if len(period_results) < 2:
            return {}
            
        # Extract key metrics for each period
        period_metrics = {}
        for period_name, period_data in period_results.items():
            metrics = {}
            
            # Autocorrelation significance
            if 'autocorrelation' in period_data:
                acf_significant = any(lag_result.get('significant', False) 
                                    for lag_result in period_data['autocorrelation'].values())
                metrics['acf_significant'] = acf_significant
            
            # ADF test
            if 'adf_price' in period_data:
                metrics['price_stationary'] = period_data['adf_price'].get('is_stationary', False)
            
            if 'adf_return' in period_data:
                metrics['return_stationary'] = period_data['adf_return'].get('is_stationary', False)
            
            # AR model
            if 'ar_model' in period_data and period_data['ar_model'] is not None:
                metrics['ar_significant'] = period_data['ar_model'].get('significant', False)
                metrics['ar_coefficient'] = period_data['ar_model'].get('ar_coefficient', 0)
                metrics['ar_r2'] = period_data['ar_model'].get('out_of_sample_r2', 0)
            
            period_metrics[period_name] = metrics
        
        # Compare early vs. late
        if 'early' in period_metrics and 'late' in period_metrics:
            early_metrics = period_metrics['early']
            late_metrics = period_metrics['late']
            
            comparison = {
                'early_more_predictable': False,
                'late_more_predictable': False,
                'no_clear_trend': True,
                'details': {}
            }
            
            # Compare autocorrelation
            if 'acf_significant' in early_metrics and 'acf_significant' in late_metrics:
                comparison['details']['acf'] = {
                    'early': early_metrics['acf_significant'],
                    'late': late_metrics['acf_significant'],
                    'change': 'More efficient' if early_metrics['acf_significant'] and not late_metrics['acf_significant'] else
                             'Less efficient' if not early_metrics['acf_significant'] and late_metrics['acf_significant'] else
                             'No change'
                }
            
            # Compare AR model significance
            if 'ar_significant' in early_metrics and 'ar_significant' in late_metrics:
                comparison['details']['ar_significant'] = {
                    'early': early_metrics['ar_significant'],
                    'late': late_metrics['ar_significant'],
                    'change': 'More efficient' if early_metrics['ar_significant'] and not late_metrics['ar_significant'] else
                             'Less efficient' if not early_metrics['ar_significant'] and late_metrics['ar_significant'] else
                             'No change'
                }
            
            # Compare AR model coefficient
            if 'ar_coefficient' in early_metrics and 'ar_coefficient' in late_metrics:
                comparison['details']['ar_coefficient'] = {
                    'early': early_metrics['ar_coefficient'],
                    'late': late_metrics['ar_coefficient'],
                    'change': 'Weaker effect' if abs(early_metrics['ar_coefficient']) > abs(late_metrics['ar_coefficient']) else
                             'Stronger effect' if abs(early_metrics['ar_coefficient']) < abs(late_metrics['ar_coefficient']) else
                             'No change'
                }
            
            # Compare AR model R²
            if 'ar_r2' in early_metrics and 'ar_r2' in late_metrics:
                comparison['details']['ar_r2'] = {
                    'early': early_metrics['ar_r2'],
                    'late': late_metrics['ar_r2'],
                    'change': 'More efficient' if early_metrics['ar_r2'] > late_metrics['ar_r2'] else
                             'Less efficient' if early_metrics['ar_r2'] < late_metrics['ar_r2'] else
                             'No change'
                }
            
            # Determine overall trend
            efficiency_indicators = [
                detail.get('change') for detail in comparison['details'].values()
                if detail.get('change') in ['More efficient', 'Less efficient']
            ]
            
            if efficiency_indicators:
                more_efficient_count = efficiency_indicators.count('More efficient')
                less_efficient_count = efficiency_indicators.count('Less efficient')
                
                if more_efficient_count > less_efficient_count:
                    comparison['early_more_predictable'] = True
                    comparison['no_clear_trend'] = False
                elif less_efficient_count > more_efficient_count:
                    comparison['late_more_predictable'] = True
                    comparison['no_clear_trend'] = False
            
            return comparison
        
        return {}
    
    def _aggregate_time_varying_results(self, market_results):
        """Aggregate time-varying efficiency results across markets."""
        if not market_results:
            return {}
            
        aggregate = {
            'total_markets': len(market_results),
            'markets_with_time_varying_efficiency': 0,
            'markets_more_efficient_over_time': 0,
            'markets_less_efficient_over_time': 0,
            'markets_no_clear_trend': 0,
            'by_election_type': {},
            'by_country': {}
        }
        
        for market_id, results in market_results.items():
            evolution = results.get('efficiency_evolution', {})
            
            if evolution:
                if not evolution.get('no_clear_trend', True):
                    aggregate['markets_with_time_varying_efficiency'] += 1
                    
                    if evolution.get('early_more_predictable', False):
                        aggregate['markets_more_efficient_over_time'] += 1
                    elif evolution.get('late_more_predictable', False):
                        aggregate['markets_less_efficient_over_time'] += 1
                else:
                    aggregate['markets_no_clear_trend'] += 1
            
            # Group by election type
            election_type = results.get('event_type', 'Unknown')
            if election_type not in aggregate['by_election_type']:
                aggregate['by_election_type'][election_type] = {
                    'count': 0,
                    'more_efficient_over_time': 0,
                    'less_efficient_over_time': 0,
                    'no_clear_trend': 0
                }
            
            aggregate['by_election_type'][election_type]['count'] += 1
            
            if evolution:
                if evolution.get('early_more_predictable', False):
                    aggregate['by_election_type'][election_type]['more_efficient_over_time'] += 1
                elif evolution.get('late_more_predictable', False):
                    aggregate['by_election_type'][election_type]['less_efficient_over_time'] += 1
                else:
                    aggregate['by_election_type'][election_type]['no_clear_trend'] += 1
            
            # Group by country
            country = results.get('country', 'Unknown')
            if country not in aggregate['by_country']:
                aggregate['by_country'][country] = {
                    'count': 0,
                    'more_efficient_over_time': 0,
                    'less_efficient_over_time': 0,
                    'no_clear_trend': 0
                }
            
            aggregate['by_country'][country]['count'] += 1
            
            if evolution:
                if evolution.get('early_more_predictable', False):
                    aggregate['by_country'][country]['more_efficient_over_time'] += 1
                elif evolution.get('late_more_predictable', False):
                    aggregate['by_country'][country]['less_efficient_over_time'] += 1
                else:
                    aggregate['by_country'][country]['no_clear_trend'] += 1
        
        # Calculate percentages
        if aggregate['total_markets'] > 0:
            aggregate['time_varying_percentage'] = aggregate['markets_with_time_varying_efficiency'] / aggregate['total_markets'] * 100
            aggregate['more_efficient_percentage'] = aggregate['markets_more_efficient_over_time'] / aggregate['total_markets'] * 100
            aggregate['less_efficient_percentage'] = aggregate['markets_less_efficient_over_time'] / aggregate['total_markets'] * 100
            aggregate['no_trend_percentage'] = aggregate['markets_no_clear_trend'] / aggregate['total_markets'] * 100
            
            # Calculate percentages for election types
            for election_type in aggregate['by_election_type']:
                count = aggregate['by_election_type'][election_type]['count']
                if count > 0:
                    aggregate['by_election_type'][election_type]['more_efficient_percentage'] = (
                        aggregate['by_election_type'][election_type]['more_efficient_over_time'] / count * 100
                    )
                    aggregate['by_election_type'][election_type]['less_efficient_percentage'] = (
                        aggregate['by_election_type'][election_type]['less_efficient_over_time'] / count * 100
                    )
                    aggregate['by_election_type'][election_type]['no_trend_percentage'] = (
                        aggregate['by_election_type'][election_type]['no_clear_trend'] / count * 100
                    )
            
            # Calculate percentages for countries
            for country in aggregate['by_country']:
                count = aggregate['by_country'][country]['count']
                if count > 0:
                    aggregate['by_country'][country]['more_efficient_percentage'] = (
                        aggregate['by_country'][country]['more_efficient_over_time'] / count * 100
                    )
                    aggregate['by_country'][country]['less_efficient_percentage'] = (
                        aggregate['by_country'][country]['less_efficient_over_time'] / count * 100
                    )
                    aggregate['by_country'][country]['no_trend_percentage'] = (
                        aggregate['by_country'][country]['no_clear_trend'] / count * 100
                    )
        
        return aggregate
    
    def generate_visualizations(self):
        """Generate summary visualizations for the efficiency analysis results."""
        if not any(self.results.values()):
            print("No results to visualize. Run the analysis first.")
            return
        
        # Create directory for visualizations
        viz_dir = f"{self.results_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Weak-form efficiency summary
        if self.results.get('weak_form'):
            self._visualize_weak_form_results(viz_dir)
        
        # 2. Cross-market predictability summary
        if self.results.get('cross_market'):
            self._visualize_cross_market_results(viz_dir)
        
        # 3. Time-varying efficiency summary
        if self.results.get('time_varying'):
            self._visualize_time_varying_results(viz_dir)
    
    def _visualize_weak_form_results(self, viz_dir):
        """Generate visualizations for weak-form efficiency results."""
        results = self.results.get('weak_form', {})
        
        # 1. Summary of efficiency tests
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = [
            ('Significant Autocorrelation', results.get('autocorrelation', {}).get('significant_percentage', 0)),
            ('Non-Stationary Prices', 100 - results.get('adf_price', {}).get('stationary_percentage', 0)),
            ('Significant AR Model', results.get('ar_model', {}).get('significant_percentage', 0)),
            ('Mean Reversion Markets', results.get('variance_ratio', {}).get('mean_reversion_markets', 0) / 
                                     results.get('total_markets', 1) * 100),
            ('Momentum Markets', results.get('variance_ratio', {}).get('momentum_markets', 0) / 
                               results.get('total_markets', 1) * 100)
        ]
        
        labels = [metric[0] for metric in metrics]
        values = [metric[1] for metric in metrics]
        
        bars = ax.barh(labels, values, color='skyblue')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    ha='left', va='center')
        
        ax.set_xlabel('Percentage of Markets')
        ax.set_title('Indicators of Market Inefficiency', fontsize=14)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/weak_form_summary.png", dpi=300)
        plt.close()
        
        # 2. Efficiency by election type
        if 'by_election_type' in results:
            election_types = list(results['by_election_type'].keys())
            significant_ar = [results['by_election_type'][et].get('significant_ar_pct', 0) 
                           for et in election_types]
            
            if len(election_types) > 1:  # Only plot if we have multiple types
                fig, ax = plt.subplots(figsize=(12, 6))
                
                y_pos = range(len(election_types))
                bars = ax.barh(y_pos, significant_ar, color='lightgreen')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                            ha='left', va='center')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(election_types)
                ax.set_xlabel('Percentage of Markets with Significant AR Model')
                ax.set_title('Market Predictability by Election Type', fontsize=14)
                ax.set_xlim(0, 100)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/weak_form_by_election_type.png", dpi=300)
                plt.close()
        
        # 3. Efficiency by country
        if 'by_country' in results:
            countries = list(results['by_country'].keys())
            # Only include countries with at least 5 markets
            countries = [c for c in countries if results['by_country'][c].get('count', 0) >= 5]
            
            significant_ar = [results['by_country'][c].get('significant_ar_pct', 0) 
                           for c in countries]
            
            if len(countries) > 1:  # Only plot if we have multiple countries
                fig, ax = plt.subplots(figsize=(12, 6))
                
                y_pos = range(len(countries))
                bars = ax.barh(y_pos, significant_ar, color='coral')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                            ha='left', va='center')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(countries)
                ax.set_xlabel('Percentage of Markets with Significant AR Model')
                ax.set_title('Market Predictability by Country', fontsize=14)
                ax.set_xlim(0, 100)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/weak_form_by_country.png", dpi=300)
                plt.close()
    
    def _visualize_cross_market_results(self, viz_dir):
        """Generate visualizations for cross-market predictability results."""
        results = self.results.get('cross_market', {})
        
        if not results:
            return
        
        # Create a summary chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = [
            ('Events with Predictability', results.get('events_with_predictability_percentage', 0)),
            ('Significant Pairs', results.get('significant_percentage', 0)),
            ('Bidirectional Causality', results.get('bidirectional_percentage', 0))
        ]
        
        labels = [metric[0] for metric in metrics]
        values = [metric[1] for metric in metrics]
        
        bars = ax.barh(labels, values, color='lightblue')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    ha='left', va='center')
        
        ax.set_xlabel('Percentage')
        ax.set_title('Cross-Market Predictability Metrics', fontsize=14)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/cross_market_summary.png", dpi=300)
        plt.close()
    
    def _visualize_time_varying_results(self, viz_dir):
        """Generate visualizations for time-varying efficiency results."""
        results = self.results.get('time_varying', {})
        
        if not results:
            return
        
        # 1. Overall efficiency changes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['More Efficient Over Time', 'Less Efficient Over Time', 'No Clear Trend']
        values = [
            results.get('more_efficient_percentage', 0),
            results.get('less_efficient_percentage', 0),
            results.get('no_trend_percentage', 0)
        ]
        
        bars = ax.bar(categories, values, color=['green', 'red', 'gray'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}%", 
                    ha='center', va='bottom')
        
        ax.set_ylabel('Percentage of Markets')
        ax.set_title('Efficiency Changes Over Market Lifetime', fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/time_varying_summary.png", dpi=300)
        plt.close()
        
        # 2. Efficiency changes by election type
        if 'by_election_type' in results:
            election_types = []
            more_efficient = []
            less_efficient = []
            
            for et, data in results['by_election_type'].items():
                if data.get('count', 0) >= 5:  # Only include types with enough data
                    election_types.append(et)
                    more_efficient.append(data.get('more_efficient_percentage', 0))
                    less_efficient.append(data.get('less_efficient_percentage', 0))
            
            if len(election_types) > 1:  # Only plot if we have multiple types
                fig, ax = plt.subplots(figsize=(12, 6))
                
                x = np.arange(len(election_types))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, more_efficient, width, label='More Efficient', color='green')
                bars2 = ax.bar(x + width/2, less_efficient, width, label='Less Efficient', color='red')
                
                ax.set_xlabel('Election Type')
                ax.set_ylabel('Percentage of Markets')
                ax.set_title('Efficiency Changes by Election Type', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(election_types, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/time_varying_by_election_type.png", dpi=300)
                plt.close()
    
    def run_analysis(self, max_markets=100, max_events=20):
        """
        Run the complete market efficiency analysis.
        
        Parameters:
        -----------
        max_markets : int
            Maximum number of markets to analyze for weak-form and time-varying tests
        max_events : int
            Maximum number of events to analyze for cross-market tests
            
        Returns:
        --------
        dict
            Dictionary with all analysis results
        """
        print("Starting market efficiency analysis...")
        
        print("\n1. Analyzing weak-form efficiency...")
        weak_form_results = self.analyze_weak_form_efficiency(max_markets=max_markets)
        print(f"  Analyzed {len(weak_form_results)} markets")
        
        print("\n2. Analyzing cross-market predictability...")
        cross_market_results = self.analyze_cross_market_predictability(max_events=max_events)
        print(f"  Analyzed {len(cross_market_results)} events")
        
        print("\n3. Analyzing time-varying efficiency...")
        time_varying_results = self.analyze_time_varying_efficiency(max_markets=max_markets//2)
        print(f"  Analyzed {len(time_varying_results)} markets")
        
        print("\n4. Generating visualizations...")
        self.generate_visualizations()
        
        print("\nAnalysis complete! Results and visualizations saved to:", self.results_dir)
        
        # Save results to JSON
        import json
        
        # Function to convert NumPy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj
        
        # Create a summary of results for JSON
        summary_results = {
            'weak_form': self.results.get('weak_form', {}),
            'cross_market': self.results.get('cross_market', {}),
            'time_varying': self.results.get('time_varying', {})
        }
        
        # Save summary to JSON
        with open(f"{self.results_dir}/market_efficiency_results.json", 'w') as f:
            json.dump(summary_results, f, default=convert_for_json, indent=2)
        
        return self.results


def main():
    """Run the market efficiency analysis as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze market efficiency in prediction markets')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--results_dir', type=str, default='results/knowledge_value/efficiency', 
                        help='Path to save results')
    parser.add_argument('--max_markets', type=int, default=100, 
                        help='Maximum number of markets to analyze')
    parser.add_argument('--max_events', type=int, default=20, 
                        help='Maximum number of events to analyze')
    
    args = parser.parse_args()
    
    analyzer = MarketEfficiencyAnalyzer(data_dir=args.data_dir, results_dir=args.results_dir)
    analyzer.run_analysis(max_markets=args.max_markets, max_events=args.max_events)


if __name__ == "__main__":
    main()