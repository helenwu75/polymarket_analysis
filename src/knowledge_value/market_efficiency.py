import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from functools import lru_cache
import warnings
import json

class MarketEfficiencyAnalyzer:
    """
    Enhanced analyzer for market efficiency in prediction markets.
    Provides tools to analyze individual markets or groups of markets.
    
    Features:
    - Efficient data caching
    - Progress tracking
    - Enhanced visualizations
    - Comprehensive efficiency metrics
    """
    
    def __init__(self, data_dir='data', results_dir='results/knowledge_value/efficiency', 
                 max_cache_size=50):
        """
        Initialize the analyzer with data paths and caching parameters.
        
        Parameters:
        -----------
        data_dir : str
            Path to the data directory
        results_dir : str
            Path to save results and plots
        max_cache_size : int
            Maximum number of markets to cache
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.max_cache_size = max_cache_size
        self.market_data_cache = {}
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load helpers for data access
        sys.path.append('../src')
        from utils.data_loader import load_main_dataset, load_market_question_mapping
        
        # Load main dataset
        try:
            self.main_df = load_main_dataset(f"{data_dir}/cleaned_election_data.csv")
            print(f"Loaded dataset with {self.main_df.shape[0]} rows and {self.main_df.shape[1]} columns")
            
            # Determine ID column
            self.id_column = None
            if 'market_id' in self.main_df.columns:
                self.id_column = 'market_id'
            elif 'id' in self.main_df.columns:
                self.id_column = 'id'
            else:
                self.id_column = self.main_df.columns[0]
                print(f"Warning: Using {self.id_column} as market ID column")
            
            # Load market questions
            self.market_questions = load_market_question_mapping(f"{data_dir}/trades/market_id_to_question.json")
            print(f"Loaded mapping for {len(self.market_questions)} markets")
            
        except Exception as e:
            warnings.warn(f"Error initializing analyzer: {e}")
            self.main_df = None
            self.market_questions = {}
    
    @lru_cache(maxsize=100)
    def preprocess_market_data(self, market_id, resample='1min', verbose=False):
        """
        Preprocess market data with efficient caching.
        
        Parameters:
        -----------
        market_id : str
            The ID of the market to analyze
        resample : str
            Frequency to resample the time series (default: '1min')
        verbose : bool
            Whether to print verbose output
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: timestamp, price, log_return
        """
        # Check if already in cache
        cache_key = (str(market_id), resample)
        if cache_key in self.market_data_cache:
            if verbose:
                print(f"Using cached data for market {market_id}")
            return self.market_data_cache[cache_key]
        
        # If cache is full, remove oldest entry
        if len(self.market_data_cache) >= self.max_cache_size:
            self.market_data_cache.pop(next(iter(self.market_data_cache)))
        
        # Load trade data
        sys.path.append('../src')
        from utils.data_loader import load_trade_data
        
        trades_df = load_trade_data(market_id, trades_dir=f"{self.data_dir}/trades")
        
        if trades_df is None or len(trades_df) < 30:
            if verbose:
                print(f"Insufficient trade data for market {market_id}")
            return None
        
        # Ensure timestamp is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            # First convert to numeric to avoid FutureWarning
            if pd.api.types.is_string_dtype(trades_df['timestamp']):
                numeric_timestamps = pd.to_numeric(trades_df['timestamp'], errors='coerce')
                trades_df['timestamp'] = pd.to_datetime(numeric_timestamps, unit='s')
            else:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='s')
        
        # Sort by timestamp
        trades_df = trades_df.sort_values('timestamp')
        
        # Ensure price is numeric - find the right column
        price_col = None
        for col in ['price', 'price_num']:
            if col in trades_df.columns:
                trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
                price_col = col
                break
        
        if price_col is None:
            if verbose:
                print(f"No price column found for market {market_id}")
            return None
        
        if verbose:
            print(f"Market ID: {market_id}")
            print(f"Original trades count: {len(trades_df)}")
            print(f"Price column: {trades_df[price_col].describe()}")
            print(f"Timestamp range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
        
        # Drop rows with NaN prices
        trades_df = trades_df.dropna(subset=[price_col])
        
        # Use price_col for consistency
        if price_col != 'price':
            trades_df['price'] = trades_df[price_col]
        
        # Resample to regular intervals with indexing optimization
        trades_df = trades_df.set_index('timestamp')
        
        # Only keep the price column to reduce memory usage
        price_series = trades_df['price']
        
        # Use efficient resampling with forward fill
        try:
            price_series = price_series.resample(resample).last().ffill()
        except Exception as e:
            if verbose:
                print(f"Error resampling: {e}")
            # Try a simpler approach
            price_series = price_series.resample(resample).mean()
        
        # Check for adequate data after resampling
        if len(price_series) < 10:
            if verbose:
                print(f"Insufficient data after resampling for market {market_id}")
            return None
        
        # Calculate log returns - avoid duplicate calculations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_returns = np.log(price_series / price_series.shift(1))
        
        # Create DataFrame efficiently
        result_df = pd.DataFrame({
            'price': price_series,
            'log_return': log_returns
        })
        
        # Drop NaN values (first row will have NaN log return)
        result_df = result_df.dropna()
        
        # Cache result
        self.market_data_cache[cache_key] = result_df
        
        return result_df
    
    def get_market_details(self, market_id):
        """
        Get detailed information about a market.
        
        Parameters:
        -----------
        market_id : str or int
            Market identifier
            
        Returns:
        --------
        dict
            Dictionary of market details
        """
        # Retrieve question directly using market ID
        question = self.market_questions.get(str(market_id), 'Unknown Question')
        
        market_details = {
            'id': market_id,
            'question': question,
            'event_type': 'Unknown',
            'country': 'Unknown',
            'volume': 0,
            'duration_days': 0
        }
        
        # Try to find market details in the main dataframe
        try:
            # Convert market_id to string for comparison
            market_row = self.main_df[self.main_df[self.id_column].astype(str) == str(market_id)]
            
            if not market_row.empty:
                market_row = market_row.iloc[0]
                
                # Try to get additional details from the dataset
                detail_mappings = [
                    ('event_electionType', 'event_type'),
                    ('event_country', 'country'),
                    ('volumeNum', 'volume'),
                    ('market_duration_days', 'duration_days')
                ]
                
                for source_col, detail_key in detail_mappings:
                    if source_col in market_row.index and not pd.isna(market_row[source_col]):
                        market_details[detail_key] = market_row[source_col]
        
        except Exception as e:
            print(f"Error retrieving market details: {e}")
        
        return market_details
    
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = adfuller(series.dropna())
                
                # Format results
                adf_result = {
                    'adf_statistic': result[0],
                    'pvalue': result[1],
                    'critical_values': result[4],
                    'is_stationary': result[1] < 0.05  # Reject unit root if p-value < 0.05
                }
                
                return adf_result
            except Exception as e:
                print(f"Error in ADF test: {e}")
                return {
                    'adf_statistic': None,
                    'pvalue': None,
                    'critical_values': None,
                    'is_stationary': None,
                    'error': str(e)
                }
    
    def run_autocorrelation_tests(self, returns, lags=10):
        """
        Run ACF tests on return series.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        lags : int
            Number of lags to test
            
        Returns:
        --------
        dict
            Dictionary with ACF results and significance
        """
        try:
            # Calculate ACF
            acf_values = acf(returns, nlags=lags, fft=True)
            
            # Calculate significance threshold
            significance_level = 1.96 / np.sqrt(len(returns))  # 95% confidence level
            
            # Check for significant autocorrelation
            significant_lags = []
            for i in range(1, len(acf_values)):  # Skip lag 0 (always 1)
                if abs(acf_values[i]) > significance_level:
                    significant_lags.append(i)
            
            result = {
                'acf_values': acf_values.tolist(),
                'significant_lags': significant_lags,
                'has_significant_autocorrelation': len(significant_lags) > 0
            }
            
            return result
        except Exception as e:
            print(f"Error in autocorrelation test: {e}")
            return {
                'error': str(e),
                'has_significant_autocorrelation': None
            }
    
    def run_variance_ratio_test(self, returns, periods=[1, 5, 15, 60]):
        """
        Run variance ratio test to check if variance scales linearly with time.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        periods : list
            List of periods to test
            
        Returns:
        --------
        dict
            Dictionary with variance ratio results
        """
        results = {}
        
        try:
            # Calculate variance for base period
            base_period = periods[0]
            base_var = returns.var()
            
            for period in periods[1:]:
                # Skip if we don't have enough data
                if len(returns) < period * 10:
                    continue
                    
                # Aggregate returns for longer period
                agg_returns = returns.rolling(window=period).sum()
                agg_returns = agg_returns.dropna()
                
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
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                results[f"{period}min"] = {
                    'variance_ratio': var_ratio,
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': 'Mean Reversion' if var_ratio < 1 else 'Momentum' if var_ratio > 1 else 'Random Walk'
                }
        except Exception as e:
            print(f"Error in variance ratio test: {e}")
            results['error'] = str(e)
        
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
        try:
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
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_statistic': z_stat,
                'p_value': p_value,
                'is_random': p_value >= 0.05  # Null hypothesis is randomness
            }
        except Exception as e:
            print(f"Error in runs test: {e}")
            return {
                'error': str(e),
                'is_random': None
            }
    
    def fit_ar_model(self, returns, lags=1):
        """
        Fit AR model to return series and evaluate predictability.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
        lags : int
            Order of the AR model
            
        Returns:
        --------
        dict
            Dictionary with model results
        """
        if len(returns) <= lags + 2:
            return None
            
        # Fit AR model
        try:
            model = AutoReg(returns, lags=lags)
            model_fit = model.fit()
            
            # Extract coefficient and p-value
            coef = model_fit.params[1] if len(model_fit.params) > 1 else 0
            p_value = model_fit.pvalues[1] if len(model_fit.pvalues) > 1 else 1
            
            return {
                'ar_coefficient': coef,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        except Exception as e:
            print(f"Error fitting AR model: {e}")
            return None
    
    def analyze_time_varying_efficiency(self, returns):
        """
        Analyze how efficiency changes over time by dividing the returns series 
        into early, middle, and late periods.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of log returns
            
        Returns:
        --------
        dict
            Dictionary with time-varying efficiency results
        """
        if len(returns) < 90:  # Need enough data to divide
            return None
        
        try:
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
                
                # Calculate autocorrelation
                acf_result = self.run_autocorrelation_tests(period_returns)
                
                # Fit AR model
                ar_result = self.fit_ar_model(period_returns)
                
                results[period_name] = {
                    'significant_acf': acf_result.get('has_significant_autocorrelation', False),
                    'ar_model': ar_result,
                    'return_volatility': period_returns.std(),
                    'sample_size': len(period_returns)
                }
            
            # Compare early vs late
            if 'early' in results and 'late' in results:
                early_ar_sig = results['early'].get('ar_model', {}).get('significant', False) if results['early'].get('ar_model') else False
                late_ar_sig = results['late'].get('ar_model', {}).get('significant', False) if results['late'].get('ar_model') else False
                
                efficiency_change = 'No Change'
                if early_ar_sig and not late_ar_sig:
                    efficiency_change = 'More Efficient'
                elif not early_ar_sig and late_ar_sig:
                    efficiency_change = 'Less Efficient'
                
                volatility_ratio = results['late']['return_volatility'] / results['early']['return_volatility'] if results['early']['return_volatility'] > 0 else 1
                
                results['comparison'] = {
                    'efficiency_change': efficiency_change,
                    'volatility_ratio': volatility_ratio,
                    'early_more_inefficient': early_ar_sig and not late_ar_sig,
                    'late_more_inefficient': not early_ar_sig and late_ar_sig
                }
            
            return results
        except Exception as e:
            print(f"Error in time-varying efficiency analysis: {e}")
            return None
    
    def analyze_market(self, market_id, verbose=False):
        """
        Run comprehensive efficiency analysis on a single market.
        
        Parameters:
        -----------
        market_id : str or int
            Market ID to analyze
        verbose : bool
            Whether to print verbose output
            
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        result = {'market_id': market_id}
        
        # Get market details
        market_details = self.get_market_details(market_id)
        result.update(market_details)
        
        if verbose:
            print(f"\nAnalyzing market: {market_details['question']}")
        
        # Preprocess market data
        market_data = self.preprocess_market_data(market_id, verbose=verbose)
        if market_data is None:
            if verbose:
                print(f"Could not process data for market {market_id}")
            return result
        
        # Run efficiency tests
        result['adf_price'] = self.run_adf_test(market_data['price'], 'price')
        result['adf_return'] = self.run_adf_test(market_data['log_return'], 'return')
        result['autocorrelation'] = self.run_autocorrelation_tests(market_data['log_return'])
        result['variance_ratio'] = self.run_variance_ratio_test(market_data['log_return'])
        result['runs_test'] = self.run_runs_test(market_data['log_return'])
        result['ar_model'] = self.fit_ar_model(market_data['log_return'])
        result['time_varying'] = self.analyze_time_varying_efficiency(market_data['log_return'])
        
        # Calculate efficiency score
        result['efficiency_score'] = self.calculate_efficiency_score(result)
        
        # Determine efficiency class
        if result['efficiency_score'] >= 80:
            result['efficiency_class'] = 'Highly Efficient'
        elif result['efficiency_score'] >= 60:
            result['efficiency_class'] = 'Moderately Efficient'
        elif result['efficiency_score'] >= 40:
            result['efficiency_class'] = 'Slightly Inefficient'
        else:
            result['efficiency_class'] = 'Highly Inefficient'
        
        if verbose:
            self.print_efficiency_summary(result)
        
        return result
    
    def calculate_efficiency_score(self, market_result):
        """
        Calculate an efficiency score based on various test results.
        
        Parameters:
        -----------
        market_result : dict
            Dictionary with market efficiency test results
            
        Returns:
        --------
        float
            Efficiency score (0-100, higher = more efficient)
        """
        score = 0
        max_points = 0
        
        # 1. Non-stationary price (random walk) = efficient
        if 'adf_price' in market_result:
            max_points += 1
            if not market_result['adf_price'].get('is_stationary', True):
                score += 1
        
        # 2. Stationary returns = efficient
        if 'adf_return' in market_result:
            max_points += 1
            if market_result['adf_return'].get('is_stationary', False):
                score += 1
        
        # 3. No significant autocorrelation = efficient
        if 'autocorrelation' in market_result:
            max_points += 1
            if not market_result['autocorrelation'].get('has_significant_autocorrelation', True):
                score += 1
        
        # 4. Random runs test = efficient
        if 'runs_test' in market_result:
            max_points += 1
            if market_result['runs_test'].get('is_random', False):
                score += 1
        
        # 5. No significant AR model = efficient
        if 'ar_model' in market_result and market_result['ar_model']:
            max_points += 1
            if not market_result['ar_model'].get('significant', True):
                score += 1
        
        # 6. Variance ratio close to 1 = efficient
        if 'variance_ratio' in market_result and market_result['variance_ratio']:
            vr_count = 0
            vr_score = 0
            for period, result in market_result['variance_ratio'].items():
                if period != 'error':  # Skip error entry
                    max_points += 0.5
                    vr_count += 0.5
                    if not result.get('significant', True):  # Not significantly different from 1
                        score += 0.5
                        vr_score += 0.5
        
        # Calculate percentage
        if max_points > 0:
            efficiency_score = (score / max_points) * 100
        else:
            efficiency_score = 0
        
        return efficiency_score
    
    def print_efficiency_summary(self, result):
        """
        Print a summary of market efficiency results.
        
        Parameters:
        -----------
        result : dict
            Dictionary with market efficiency results
        """
        print("\n" + "="*60)
        print(f"MARKET EFFICIENCY SUMMARY")
        print("="*60)
        print(f"Market: {result.get('question', 'Unknown')}")
        print(f"Market ID: {result.get('market_id', 'Unknown')}")
        print(f"Type: {result.get('event_type', 'Unknown')}")
        print(f"Country: {result.get('country', 'Unknown')}")
        print("-"*60)
        
        print("\nEfficiency Tests:")
        
        def print_test_result(name, result_dict, key, value_true, value_false):
            if result_dict is None:
                print(f"  {name}: N/A")
                return
            
            value = result_dict.get(key)
            if value is None:
                print(f"  {name}: N/A")
            else:
                print(f"  {name}: {value_true if value else value_false}")
        
        print_test_result("Price Process", result.get('adf_price'), 'is_stationary', 
                         "Stationary (Inefficient)", "Non-stationary - Random Walk (Efficient)")
        
        print_test_result("Returns", result.get('adf_return'), 'is_stationary', 
                         "Stationary (Efficient)", "Non-stationary (Inefficient)")
        
        print_test_result("Autocorrelation", result.get('autocorrelation'), 'has_significant_autocorrelation', 
                         "Significant (Inefficient)", "Not Significant (Efficient)")
        
        print_test_result("Runs Test", result.get('runs_test'), 'is_random', 
                         "Random (Efficient)", "Non-random (Inefficient)")
        
        ar_model = result.get('ar_model')
        if ar_model:
            print(f"  AR(1) Model: {'Significant (Inefficient)' if ar_model.get('significant') else 'Not Significant (Efficient)'}")
            print(f"    Coefficient: {ar_model.get('ar_coefficient', 'N/A'):.6f}")
            print(f"    P-value: {ar_model.get('p_value', 'N/A'):.6f}")
        else:
            print("  AR(1) Model: N/A")
        
        # Variance ratio
        vr = result.get('variance_ratio', {})
        if vr and not 'error' in vr:
            print("\n  Variance Ratio Test:")
            for period, period_result in vr.items():
                if period != 'error':
                    print(f"    {period}: {period_result.get('variance_ratio', 'N/A'):.4f} "
                          f"(p-value: {period_result.get('p_value', 'N/A'):.4f}) - "
                          f"{period_result.get('interpretation', 'Unknown')}")
        
        # Time-varying efficiency
        tv = result.get('time_varying', {})
        if tv and 'comparison' in tv:
            print("\n  Time-varying Efficiency:")
            print(f"    Efficiency Change: {tv['comparison'].get('efficiency_change', 'Unknown')}")
            print(f"    Volatility Change: {(tv['comparison'].get('volatility_ratio', 1) - 1) * 100:.1f}%")
        
        print("\n" + "-"*60)
        print(f"Overall Efficiency Score: {result.get('efficiency_score', 0):.2f}/100")
        print(f"Classification: {result.get('efficiency_class', 'Unknown')}")
        print("="*60)
    
    def visualize_market(self, market_id, market_data=None, save_to=None, plot_type='static'):
        """
        Create visualizations for market data and efficiency analysis.
        
        Parameters:
        -----------
        market_id : str or int
            Market ID to visualize
        market_data : pd.DataFrame, optional
            Preprocessed market data (if None, will be loaded)
        save_to : str, optional
            Path to save the visualization
        plot_type : str
            Type of plot ('static' for matplotlib, 'interactive' for plotly)
            
        Returns:
        --------
        dict
            Dictionary with figure objects
        """
        # Get market details
        market_details = self.get_market_details(market_id)
        
        # Load data if not provided
        if market_data is None:
            market_data = self.preprocess_market_data(market_id)
            
        if market_data is None:
            print(f"Could not load data for market {market_id}")
            return None
        
        # Run efficiency tests for visualization
        acf_result = self.run_autocorrelation_tests(market_data['log_return'])
        
        figures = {}
        
        if plot_type == 'interactive':
            # Create plotly figures
            
            # 1. Price Series
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=market_data.index, 
                y=market_data['price'],
                mode='lines',
                name='Price'
            ))
            fig1.update_layout(
                title=f'Price Series: {market_details["question"]}',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white'
            )
            figures['price_series'] = fig1
            
            # 2. Log Returns
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=market_data.index, 
                y=market_data['log_return'],
                mode='lines',
                name='Log Return',
                line=dict(color='green')
            ))
            fig2.update_layout(
                title=f'Log Returns: {market_details["question"]}',
                xaxis_title='Date',
                yaxis_title='Log Return',
                template='plotly_white'
            )
            figures['log_returns'] = fig2
            
            # 3. ACF Plot
            if acf_result and 'acf_values' in acf_result:
                acf_values = acf_result['acf_values']
                lags = list(range(len(acf_values)))
                
                significance = 1.96 / np.sqrt(len(market_data))
                
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=lags, 
                    y=acf_values,
                    name='ACF'
                ))
                
                # Add confidence intervals
                fig3.add_shape(
                    type="line",
                    x0=0,
                    y0=significance,
                    x1=max(lags),
                    y1=significance,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    ),
                )
                fig3.add_shape(
                    type="line",
                    x0=0,
                    y0=-significance,
                    x1=max(lags),
                    y1=-significance,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    ),
                )
                
                fig3.update_layout(
                    title=f'Autocorrelation Function: {"Significant" if acf_result.get("has_significant_autocorrelation", False) else "Not Significant"}',
                    xaxis_title='Lag',
                    yaxis_title='ACF',
                    template='plotly_white'
                )
                figures['acf'] = fig3
            
            # 4. Price distribution
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(
                x=market_data['price'],
                opacity=0.7,
                nbinsx=30,
                name='Price'
            ))
            fig4.update_layout(
                title=f'Price Distribution: {market_details["question"]}',
                xaxis_title='Price',
                yaxis_title='Count',
                template='plotly_white'
            )
            figures['price_dist'] = fig4
            
        else:
            # Create static matplotlib figures
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 1. Combined plot: Price and Returns
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            
            # Price Series
            axes[0, 0].plot(market_data.index, market_data['price'], linewidth=2)
            axes[0, 0].set_title(f'Price Series: {market_details["question"]}', fontsize=14)
            axes[0, 0].set_xlabel('Date', fontsize=12)
            axes[0, 0].set_ylabel('Price', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Log Returns
            axes[0, 1].plot(market_data.index, market_data['log_return'], linewidth=1, color='green')
            axes[0, 1].set_title(f'Log Returns: {market_details["question"]}', fontsize=14)
            axes[0, 1].set_xlabel('Date', fontsize=12)
            axes[0, 1].set_ylabel('Log Return', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            
            # ACF Plot
            if acf_result and 'acf_values' in acf_result:
                acf_values = acf_result['acf_values']
                lags = range(len(acf_values))
                
                axes[1, 0].bar(lags, acf_values, width=0.4)
                
                # Add confidence intervals
                significance = 1.96 / np.sqrt(len(market_data))
                axes[1, 0].axhline(y=0, linestyle='-', color='black')
                axes[1, 0].axhline(y=significance, linestyle='--', color='red', alpha=0.7)
                axes[1, 0].axhline(y=-significance, linestyle='--', color='red', alpha=0.7)
                
                title = f'Autocorrelation Function: {"❌ Significant" if acf_result.get("has_significant_autocorrelation", False) else "✅ Not Significant"}'
                axes[1, 0].set_title(title, fontsize=14)
                axes[1, 0].set_xlabel('Lag', fontsize=12)
                axes[1, 0].set_ylabel('ACF', fontsize=12)
            
            # Price Distribution
            axes[1, 1].hist(market_data['price'], bins=30, alpha=0.7, density=True)
            axes[1, 1].set_title(f'Price Distribution: {market_details["question"]}', fontsize=14)
            axes[1, 1].set_xlabel('Price', fontsize=12)
            axes[1, 1].set_ylabel('Density', fontsize=12)
            
            plt.tight_layout()
            
            if save_to:
                plt.savefig(save_to, dpi=300, bbox_inches='tight')
            
            figures['combined'] = fig
        
        return figures

    def analyze_market_batch(self, market_ids, max_markets=20, parallel=True):
        """
        Analyze multiple markets efficiently.
        
        Parameters:
        -----------
        market_ids : list
            List of market IDs to analyze
        max_markets : int
            Maximum number of markets to analyze
        parallel : bool
            Whether to use parallel processing
            
        Returns:
        --------
        list
            List of market analysis results
        """
        if not market_ids:
            print("No market IDs provided")
            return []
        
        # Limit number of markets
        if len(market_ids) > max_markets:
            print(f"Limiting analysis to {max_markets} markets")
            market_ids = market_ids[:max_markets]
        
        results = []
        
        if parallel and len(market_ids) > 5:
            try:
                # Try to use parallel processing
                from concurrent.futures import ProcessPoolExecutor, as_completed
                import multiprocessing
                
                # Determine number of workers
                n_workers = min(multiprocessing.cpu_count() - 1, 4)  # Limit to 4 workers max
                n_workers = max(1, n_workers)  # At least 1 worker
                
                print(f"Analyzing {len(market_ids)} markets using {n_workers} workers...")
                
                # Create a local function that doesn't use self (for better serialization)
                def analyze_single_market(market_id):
                    analyzer = MarketEfficiencyAnalyzer(self.data_dir, self.results_dir)
                    return analyzer.analyze_market(market_id)
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit jobs
                    futures = {executor.submit(analyze_single_market, mid): mid for mid in market_ids}
                    
                    # Process results as they complete
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            print(f"Error processing market {futures[future]}: {e}")
                            
            except (ImportError, Exception) as e:
                print(f"Parallel processing failed: {e}. Falling back to sequential processing.")
                # Fall back to sequential processing
                for market_id in tqdm(market_ids, desc="Analyzing markets"):
                    try:
                        result = self.analyze_market(market_id)
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing market {market_id}: {e}")
        else:
            # Sequential processing
            for market_id in tqdm(market_ids, desc="Analyzing markets"):
                try:
                    result = self.analyze_market(market_id)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing market {market_id}: {e}")
        
        return results

    def analyze_cross_market(self, event_id, max_lag=3):
        """
        Analyze Granger causality between markets in the same event.
        
        Parameters:
        -----------
        event_id : str
            ID of the event to analyze
        max_lag : int
            Maximum lag for Granger causality test
            
        Returns:
        --------
        dict
            Dictionary with Granger causality results
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Find markets in the event
        event_markets = []
        
        # Find the event column name
        event_col = None
        for col in ['event_id', 'groupId', 'eventId']:
            if col in self.main_df.columns:
                event_col = col
                break
        
        if event_col:
            event_df = self.main_df[self.main_df[event_col] == event_id]
            if not event_df.empty:
                event_markets = event_df[self.id_column].tolist()
        
        if not event_markets:
            print(f"No markets found for event {event_id}")
            return None
        
        print(f"Found {len(event_markets)} markets in event {event_id}")
        if len(event_markets) < 2:
            print("At least 2 markets are required for cross-market analysis")
            return None
        
        # Process each market
        market_data = {}
        for market_id in tqdm(event_markets, desc="Processing markets"):
            data = self.preprocess_market_data(market_id, resample='5min')  # Use 5-min intervals for cross-market analysis
            if data is not None and len(data) > 30:
                market_info = self.get_market_details(market_id)
                market_data[market_id] = {
                    'data': data,
                    'question': market_info['question']
                }
        
        if len(market_data) < 2:
            print("Insufficient data for cross-market analysis")
            return None
        
        print(f"Analyzing relationships between {len(market_data)} markets...")
        
        results = {
            'event_id': event_id,
            'market_pairs': [],
            'significant_pairs': 0,
            'bidirectional_pairs': 0,
            'total_pairs': 0
        }
        
        # Pairwise Granger causality tests
        for i, (market_i, data_i) in enumerate(market_data.items()):
            for j, (market_j, data_j) in enumerate(market_data.items()):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                
                # Align time series
                common_index = data_i['data'].index.intersection(data_j['data'].index)
                if len(common_index) <= max_lag + 5:
                    continue
                
                series_i = data_i['data'].loc[common_index, 'price']
                series_j = data_j['data'].loc[common_index, 'price']
                
                results['total_pairs'] += 2  # Test both directions
                
                try:
                    # Test if market i Granger-causes market j
                    gc_result_ij = grangercausalitytests(
                        pd.concat([series_j, series_i], axis=1), 
                        maxlag=max_lag, 
                        verbose=False
                    )
                    
                    # Get minimum p-value across all lags
                    min_pvalue_ij = min([test[0]['ssr_chi2test'][1] for lag, test in gc_result_ij.items()])
                    significant_ij = min_pvalue_ij < 0.05
                    
                    if significant_ij:
                        results['significant_pairs'] += 1
                    
                    # Test if market j Granger-causes market i
                    gc_result_ji = grangercausalitytests(
                        pd.concat([series_i, series_j], axis=1), 
                        maxlag=max_lag, 
                        verbose=False
                    )
                    
                    min_pvalue_ji = min([test[0]['ssr_chi2test'][1] for lag, test in gc_result_ji.items()])
                    significant_ji = min_pvalue_ji < 0.05
                    
                    if significant_ji:
                        results['significant_pairs'] += 1
                    
                    # Check for bidirectional causality
                    bidirectional = significant_ij and significant_ji
                    if bidirectional:
                        results['bidirectional_pairs'] += 1
                    
                    # Store pair result
                    pair_result = {
                        'market_i_id': market_i,
                        'market_j_id': market_j,
                        'market_i_question': data_i['question'],
                        'market_j_question': data_j['question'],
                        'i_causes_j_pvalue': min_pvalue_ij,
                        'j_causes_i_pvalue': min_pvalue_ji,
                        'i_causes_j': significant_ij,
                        'j_causes_i': significant_ji,
                        'bidirectional': bidirectional,
                        'relationship': 'Bidirectional' if bidirectional else
                                    f"{market_i} → {market_j}" if significant_ij else
                                    f"{market_j} → {market_i}" if significant_ji else
                                    "No relationship"
                    }
                    
                    results['market_pairs'].append(pair_result)
                    
                except Exception as e:
                    print(f"Error in Granger causality test between {market_i} and {market_j}: {e}")
        
        # Calculate summary statistics
        if results['total_pairs'] > 0:
            results['significant_percentage'] = results['significant_pairs'] / results['total_pairs'] * 100
            results['bidirectional_percentage'] = results['bidirectional_pairs'] / (results['total_pairs'] // 2) * 100
        
        return results

    def summarize_results(self, results):
        """
        Summarize efficiency results across multiple markets.
        
        Parameters:
        -----------
        results : list
            List of market analysis results
            
        Returns:
        --------
        dict
            Dictionary with summary statistics
        """
        if not results:
            return None
        
        summary = {
            'total_markets': len(results),
            'average_efficiency_score': np.mean([r['efficiency_score'] for r in results if 'efficiency_score' in r]),
            'efficiency_classes': {},
            'by_event_type': {},
            'by_country': {},
            'test_results': {
                'price_stationary': 0,
                'return_stationary': 0,
                'has_autocorrelation': 0,
                'is_random': 0,
                'ar_significant': 0
            }
        }
        
        # Count efficiency classes
        for result in results:
            if 'efficiency_class' in result:
                cls = result['efficiency_class']
                summary['efficiency_classes'][cls] = summary['efficiency_classes'].get(cls, 0) + 1
        
        # Aggregate by event type
        for result in results:
            if 'event_type' in result:
                event_type = result['event_type']
                if event_type not in summary['by_event_type']:
                    summary['by_event_type'][event_type] = {
                        'count': 0,
                        'total_score': 0,
                        'efficiency_classes': {}
                    }
                
                summary['by_event_type'][event_type]['count'] += 1
                
                if 'efficiency_score' in result:
                    summary['by_event_type'][event_type]['total_score'] += result['efficiency_score']
                
                if 'efficiency_class' in result:
                    cls = result['efficiency_class']
                    summary['by_event_type'][event_type]['efficiency_classes'][cls] = (
                        summary['by_event_type'][event_type]['efficiency_classes'].get(cls, 0) + 1
                    )
        
        # Calculate averages for event types
        for event_type in summary['by_event_type']:
            if summary['by_event_type'][event_type]['count'] > 0:
                summary['by_event_type'][event_type]['average_score'] = (
                    summary['by_event_type'][event_type]['total_score'] / 
                    summary['by_event_type'][event_type]['count']
                )
        
        # Aggregate by country
        for result in results:
            if 'country' in result:
                country = result['country']
                if country not in summary['by_country']:
                    summary['by_country'][country] = {
                        'count': 0,
                        'total_score': 0,
                        'efficiency_classes': {}
                    }
                
                summary['by_country'][country]['count'] += 1
                
                if 'efficiency_score' in result:
                    summary['by_country'][country]['total_score'] += result['efficiency_score']
                
                if 'efficiency_class' in result:
                    cls = result['efficiency_class']
                    summary['by_country'][country]['efficiency_classes'][cls] = (
                        summary['by_country'][country]['efficiency_classes'].get(cls, 0) + 1
                    )
        
        # Calculate averages for countries
        for country in summary['by_country']:
            if summary['by_country'][country]['count'] > 0:
                summary['by_country'][country]['average_score'] = (
                    summary['by_country'][country]['total_score'] / 
                    summary['by_country'][country]['count']
                )
        
        # Count test results
        for result in results:
            if 'adf_price' in result and result['adf_price'].get('is_stationary', False):
                summary['test_results']['price_stationary'] += 1
            
            if 'adf_return' in result and result['adf_return'].get('is_stationary', False):
                summary['test_results']['return_stationary'] += 1
            
            if 'autocorrelation' in result and result['autocorrelation'].get('has_significant_autocorrelation', False):
                summary['test_results']['has_autocorrelation'] += 1
            
            if 'runs_test' in result and result['runs_test'].get('is_random', False):
                summary['test_results']['is_random'] += 1
            
            if 'ar_model' in result and result['ar_model'] and result['ar_model'].get('significant', False):
                summary['test_results']['ar_significant'] += 1
        
        # Calculate percentages for test results
        for key in summary['test_results']:
            summary['test_results'][f'{key}_percentage'] = (
                summary['test_results'][key] / summary['total_markets'] * 100
            )
        
        return summary

    def visualize_summary(self, summary, save_dir=None):
        """
        Create summary visualizations for efficiency results.
        
        Parameters:
        -----------
        summary : dict
            Summary dictionary from summarize_results
        save_dir : str, optional
            Directory to save visualization files
            
        Returns:
        --------
        dict
            Dictionary with figure objects
        """
        if not summary:
            print("No summary data to visualize")
            return None
        
        if save_dir is None:
            save_dir = self.results_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        # 1. Efficiency Class Distribution
        plt.figure(figsize=(10, 6))
        
        classes = summary['efficiency_classes']
        labels = list(classes.keys())
        counts = list(classes.values())
        
        # Sort by efficiency level
        order = {
            'Highly Efficient': 0,
            'Moderately Efficient': 1,
            'Slightly Inefficient': 2,
            'Highly Inefficient': 3
        }
        
        sorted_data = sorted(zip(labels, counts), key=lambda x: order.get(x[0], 99))
        labels, counts = zip(*sorted_data) if sorted_data else ([], [])
        
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Green to red
        
        bars = plt.bar(labels, counts, color=colors[:len(labels)])
        
        # Add percentage labels
        total = summary['total_markets']
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height/total*100:.1f}%',
                    ha='center', va='bottom')
        
        plt.title('Market Efficiency Classification Distribution', fontsize=14)
        plt.ylabel('Number of Markets', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'efficiency_class_distribution.png'), dpi=300, bbox_inches='tight')
        figures['efficiency_distribution'] = plt
        
        # 2. Efficiency by Event Type
        event_types = summary['by_event_type']
        if len(event_types) > 1:
            # Filter event types with enough data
            min_markets = 3
            filtered_event_types = {k: v for k, v in event_types.items() if v['count'] >= min_markets}
            
            if filtered_event_types:
                plt.figure(figsize=(12, 6))
                
                event_type_labels = list(filtered_event_types.keys())
                avg_scores = [filtered_event_types[et]['average_score'] for et in event_type_labels]
                counts = [filtered_event_types[et]['count'] for et in event_type_labels]
                
                # Sort by average score
                sorted_data = sorted(zip(event_type_labels, avg_scores, counts), key=lambda x: x[1], reverse=True)
                event_type_labels, avg_scores, counts = zip(*sorted_data)
                
                # Create figure with dual y-axis
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot average scores
                bars = ax1.bar(event_type_labels, avg_scores, color='#3498db')
                ax1.set_xlabel('Event Type', fontsize=12)
                ax1.set_ylabel('Average Efficiency Score', fontsize=12)
                ax1.set_ylim(0, 100)
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax1.text(bar.get_x() + bar.get_width()/2., 5,
                            f'n={count}',
                            ha='center', va='bottom',
                            color='white', fontweight='bold')
                
                # Add horizontal line for overall average
                ax1.axhline(y=summary['average_efficiency_score'], color='r', linestyle='--',
                        label=f'Overall Average ({summary["average_efficiency_score"]:.1f})')
                
                plt.title('Average Efficiency Score by Event Type', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(save_dir, 'efficiency_by_event_type.png'), dpi=300, bbox_inches='tight')
                figures['efficiency_by_event_type'] = plt
        
        # 3. Test Results Summary
        test_results = summary['test_results']
        if test_results:
            plt.figure(figsize=(10, 6))
            
            test_labels = [
                'Non-Stationary Prices', 
                'Stationary Returns',
                'No Autocorrelation',
                'Random Runs Test',
                'No AR Predictability'
            ]
            
            test_values = [
                100 - test_results.get('price_stationary_percentage', 0),
                test_results.get('return_stationary_percentage', 0),
                100 - test_results.get('has_autocorrelation_percentage', 0),
                test_results.get('is_random_percentage', 0),
                100 - test_results.get('ar_significant_percentage', 0)
            ]
            
            # Create horizontal bars
            bars = plt.barh(test_labels, test_values, color='#3498db')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%',
                    ha='left', va='center')
            
            plt.title('Efficient Market Hypothesis Tests Results', fontsize=14)
            plt.xlabel('Percentage of Markets', fontsize=12)
            plt.xlim(0, 100)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(save_dir, 'efficiency_test_results.png'), dpi=300, bbox_inches='tight')
            figures['test_results'] = plt
        
        return figures

    def get_top_markets_by_volume(self, n=10):
        """
        Get the top N markets by trading volume.
        
        Parameters:
        -----------
        n : int
            Number of markets to return
            
        Returns:
        --------
        list
            List of market IDs
        """
        if 'volumeNum' not in self.main_df.columns:
            print("Volume data not available")
            return []
        
        top_markets = self.main_df.sort_values('volumeNum', ascending=False).head(n)
        return top_markets[self.id_column].tolist()

    def get_markets_by_event_type(self, event_type, n=None):
        """
        Get markets by event type.
        
        Parameters:
        -----------
        event_type : str
            Event type to filter by
        n : int, optional
            Maximum number of markets to return
            
        Returns:
        --------
        list
            List of market IDs
        """
        if 'event_electionType' not in self.main_df.columns:
            print("Event type data not available")
            return []
        
        markets = self.main_df[self.main_df['event_electionType'] == event_type]
        
        if n is not None:
            markets = markets.head(n)
        
        return markets[self.id_column].tolist()

    def get_markets_by_country(self, country, n=None):
        """
        Get markets by country.
        
        Parameters:
        -----------
        country : str
            Country to filter by
        n : int, optional
            Maximum number of markets to return
            
        Returns:
        --------
        list
            List of market IDs
        """
        if 'event_country' not in self.main_df.columns:
            print("Country data not available")
            return []
        
        markets = self.main_df[self.main_df['event_country'] == country]
        
        if n is not None:
            markets = markets.head(n)
        
        return markets[self.id_column].tolist()
    
    def find_market_by_name(self, name_fragment):
        """
        Find markets by partial name match.
        
        Parameters:
        -----------
        name_fragment : str
            Fragment of market name to search for
            
        Returns:
        --------
        list
            List of (market_id, market_name) tuples
        """
        matches = []
        
        # Search in main_df if available
        if self.main_df is not None and 'question' in self.main_df.columns:
            mask = self.main_df['question'].str.contains(name_fragment, case=False, na=False)
            if mask.any():
                results = self.main_df[mask]
                matches.extend([(row[self.id_column], row['question']) for _, row in results.iterrows()])
        
        # Search in market_questions dictionary
        for market_id, question in self.market_questions.items():
            if name_fragment.lower() in question.lower():
                if not any(market_id == mid for mid, _ in matches):
                    matches.append((market_id, question))
        
        return matches

    def save_results(self, results, filename=None):
        """
        Save analysis results to file.
        
        Parameters:
        -----------
        results : dict or list
            Analysis results to save
        filename : str, optional
            Filename to save results to
            
        Returns:
        --------
        str
            Path to saved file
        """
        if not results:
            return None
        
        if filename is None:
            if isinstance(results, list):
                filename = 'market_efficiency_batch_results.json'
            else:
                filename = 'market_efficiency_result.json'
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy types and handle other non-serializable objects
        def json_serialize(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, bytes):
                return obj.decode('utf-8')
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            return str(obj)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, default=json_serialize, indent=2)
            print(f"Results saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving results: {e}")
            return None

    def load_results(self, filename):
        """
        Load analysis results from file.
        
        Parameters:
        -----------
        filename : str
            Filename to load results from
            
        Returns:
        --------
        dict or list
            Loaded results
        """
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            print(f"Results loaded from {filepath}")
            return results
        except Exception as e:
            print(f"Error loading results: {e}")
            return None