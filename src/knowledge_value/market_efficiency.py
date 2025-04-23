# market_efficiency_analysis.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, adfuller, grangercausalitytests
from statsmodels.tsa.ar_model import AutoReg
from tqdm.auto import tqdm
import warnings
import json

class MarketEfficiencyAnalyzer:
    """
    Analyzer for testing market efficiency in prediction markets.
    Focuses on weak-form and strong-form efficiency tests.
    """
    
    def __init__(self, data_dir='data', results_dir='results/market_efficiency'):
        """Initialize the analyzer with data paths"""
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.market_data_cache = {}
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load dataset
        self.main_df = self._load_main_dataset()
        
        # Set ID column
        self.id_column = self._determine_id_column()
        
        # Load market questions
        self.market_questions = self._load_market_questions()
    
    def _load_main_dataset(self):
        """Load main dataset with error handling"""
        try:
            filepath = os.path.join(self.data_dir, 'cleaned_election_data.csv')
            if not os.path.exists(filepath):
                print(f"Warning: Main dataset file not found at {filepath}")
                return pd.DataFrame()  # Return empty DataFrame instead of None
                
            df = pd.read_csv(filepath, low_memory=False)
            print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading main dataset: {e}")
            return pd.DataFrame()  # Return empty DataFrame
    
    def _determine_id_column(self):
        """Determine which column contains market IDs"""
        if self.main_df.empty:
            return 'id'
            
        if 'market_id' in self.main_df.columns:
            return 'market_id'
        elif 'id' in self.main_df.columns:
            return 'id'
        else:
            return self.main_df.columns[0]
    
    def _load_market_questions(self):
        """Load market question mapping with error handling"""
        try:
            filepath = os.path.join(self.data_dir, 'trades', 'market_id_to_question.json')
            if not os.path.exists(filepath):
                print(f"Warning: Market questions file not found at {filepath}")
                return {}
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"Loaded mapping for {len(data)} markets")
            return data
        except Exception as e:
            print(f"Error loading market questions: {e}")
            return {}
    
    def load_trade_data(self, market_id):
        """Load trade data for a specific market"""
        try:
            # Check if the data is in the cache
            if market_id in self.market_data_cache:
                return self.market_data_cache[market_id]
            
            # Try to find the correct file
            from utils.data_loader import load_trade_data
            trades_df = load_trade_data(market_id, trades_dir=os.path.join(self.data_dir, 'trades'))
            
            if trades_df is None or len(trades_df) < 30:
                print(f"Insufficient trade data for market {market_id}")
                return None
            
            # Process timestamps
            if 'timestamp' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                try:
                    # Convert numeric timestamps to datetime
                    trades_df['timestamp'] = pd.to_datetime(pd.to_numeric(trades_df['timestamp']), unit='s')
                except:
                    print(f"Warning: Could not convert timestamps for market {market_id}")
            
            # Process prices
            if 'price' in trades_df.columns:
                trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
            
            # Sort by timestamp
            if 'timestamp' in trades_df.columns:
                trades_df = trades_df.sort_values('timestamp')
            
            # Calculate log returns if price exists
            if 'price' in trades_df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trades_df['log_return'] = np.log(trades_df['price'] / trades_df['price'].shift(1))
            
            # Cache the data
            self.market_data_cache[market_id] = trades_df
            
            return trades_df
        except Exception as e:
            print(f"Error loading trade data for market {market_id}: {e}")
            return None
    
    def preprocess_market_data(self, market_id, resample='1min'):
        """Preprocess market data for efficiency tests"""
        trades_df = self.load_trade_data(market_id)
        
        if trades_df is None:
            return None
        
        try:
            # Set timestamp as index
            if 'timestamp' in trades_df.columns:
                trades_df = trades_df.set_index('timestamp')
            
            # Resample data to regular intervals
            if 'price' in trades_df.columns:
                price_series = trades_df['price']
                price_series = price_series.resample(resample).last().dropna()
                
                # Calculate log returns
                log_returns = np.log(price_series / price_series.shift(1)).dropna()
                
                # Create output DataFrame
                result_df = pd.DataFrame({
                    'price': price_series,
                    'log_return': log_returns
                })
                
                return result_df
            else:
                print(f"No price data for market {market_id}")
                return None
        except Exception as e:
            print(f"Error preprocessing data for market {market_id}: {e}")
            return None
    
    def get_market_details(self, market_id):
        """Get details about a specific market"""
        question = self.market_questions.get(str(market_id), 'Unknown')
        
        details = {
            'id': market_id,
            'question': question
        }
        
        # Add more details from main_df if available
        if not self.main_df.empty:
            try:
                row = self.main_df[self.main_df[self.id_column] == market_id]
                if not row.empty:
                    for col in ['event_electionType', 'event_country', 'volumeNum']:
                        if col in row.columns:
                            details[col.replace('event_', '')] = row[col].iloc[0]
            except Exception as e:
                print(f"Error getting market details: {e}")
        
        return details
    
    def find_market_by_name(self, search_term):
        """Find markets by name or keyword"""
        matches = []
        
        # Search in main_df
        if not self.main_df.empty and 'question' in self.main_df.columns:
            matches_df = self.main_df[self.main_df['question'].str.contains(search_term, case=False, na=False)]
            for _, row in matches_df.iterrows():
                matches.append((row[self.id_column], row['question']))
        
        # Search in market_questions
        for market_id, question in self.market_questions.items():
            if search_term.lower() in question.lower():
                # Avoid duplicates
                if not any(str(market_id) == str(m_id) for m_id, _ in matches):
                    matches.append((market_id, question))
        
        return matches
    
    # Efficiency Tests
    
    def run_adf_test(self, series):
        """Run Augmented Dickey-Fuller test for stationarity"""
        try:
            result = adfuller(series.dropna())
            return {
                'statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            print(f"Error in ADF test: {e}")
            return None
    
    def run_autocorrelation_test(self, series, lags=10):
        """Test for autocorrelation in returns"""
        try:
            acf_values = acf(series.dropna(), nlags=lags, fft=True)
            
            # Calculate significance threshold
            n = len(series.dropna())
            significance = 1.96 / np.sqrt(n)
            
            # Check which lags have significant autocorrelation
            significant_lags = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > significance]
            
            return {
                'acf_values': acf_values.tolist(),
                'significant_lags': significant_lags,
                'has_significant_autocorrelation': len(significant_lags) > 0
            }
        except Exception as e:
            print(f"Error in autocorrelation test: {e}")
            return None
    
    def fit_ar_model(self, series, lags=1):
        """Fit autoregressive model to test predictability"""
        try:
            model = AutoReg(series.dropna(), lags=lags)
            model_fit = model.fit()
            
            # Extract coefficient and p-value
            coef = model_fit.params[1] if len(model_fit.params) > 1 else 0
            p_value = model_fit.pvalues[1] if len(model_fit.pvalues) > 1 else 1
            
            return {
                'coefficient': coef,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        except Exception as e:
            print(f"Error in AR model: {e}")
            return None
    
    def run_variance_ratio_test(self, returns, periods=[1, 5, 10]):
        """Run variance ratio test (Lo-MacKinlay)"""
        results = {}
        
        try:
            base_period = periods[0]
            base_var = returns.var()
            
            for period in periods[1:]:
                if len(returns) < period * 5:
                    continue
                
                # Aggregate returns for longer period
                agg_returns = returns.rolling(window=period).sum().dropna()
                period_var = agg_returns.var()
                
                # Calculate variance ratio
                var_ratio = period_var / (period * base_var)
                
                # Interpretation
                if 0.95 < var_ratio < 1.05:
                    interpretation = "Random Walk"
                elif var_ratio < 0.95:
                    interpretation = "Mean Reversion"
                else:
                    interpretation = "Momentum"
                
                results[f"period_{period}"] = {
                    'variance_ratio': var_ratio,
                    'interpretation': interpretation
                }
        except Exception as e:
            print(f"Error in variance ratio test: {e}")
        
        return results
    
    def analyze_time_varying_efficiency(self, returns):
        """Analyze how efficiency changes over time"""
        if len(returns) < 60:
            return None
        
        try:
            # Divide into three periods
            period_size = len(returns) // 3
            periods = {
                'early': returns.iloc[:period_size],
                'middle': returns.iloc[period_size:2*period_size],
                'late': returns.iloc[2*period_size:]
            }
            
            results = {}
            
            for name, period_returns in periods.items():
                # Run tests on each period
                acf_test = self.run_autocorrelation_test(period_returns)
                ar_test = self.fit_ar_model(period_returns)
                
                results[name] = {
                    'has_autocorrelation': acf_test['has_significant_autocorrelation'] if acf_test else None,
                    'ar_significant': ar_test['is_significant'] if ar_test else None,
                    'volatility': period_returns.std()
                }
            
            # Determine if efficiency changed
            if results['early']['ar_significant'] and not results['late']['ar_significant']:
                efficiency_change = "More Efficient"
            elif not results['early']['ar_significant'] and results['late']['ar_significant']:
                efficiency_change = "Less Efficient"
            else:
                efficiency_change = "No Change"
            
            results['summary'] = {
                'efficiency_change': efficiency_change,
                'volatility_change': (results['late']['volatility'] / results['early']['volatility']) - 1
            }
            
            return results
        except Exception as e:
            print(f"Error in time-varying efficiency analysis: {e}")
            return None
    
    def analyze_market(self, market_id, verbose=False):
        """Run comprehensive efficiency analysis on a market"""
        # Get market details
        details = self.get_market_details(market_id)
        
        if verbose:
            print(f"\nAnalyzing market: {details['question']}")
        
        # Preprocess data
        market_data = self.preprocess_market_data(market_id)
        if market_data is None:
            if verbose:
                print(f"Could not process data for market {market_id}")
            return {'id': market_id, 'question': details['question'], 'analysis_success': False}
        
        results = {
            'id': market_id,
            'question': details['question'],
            'analysis_success': True,
            'data_points': len(market_data)
        }
        
        # Add other details if available
        for key in ['electionType', 'country', 'volumeNum']:
            if key in details:
                results[key] = details[key]
        
        # Run weak-form efficiency tests
        results['adf_price'] = self.run_adf_test(market_data['price'])
        results['adf_return'] = self.run_adf_test(market_data['log_return'])
        results['autocorrelation'] = self.run_autocorrelation_test(market_data['log_return'])
        results['ar_model'] = self.fit_ar_model(market_data['log_return'])
        results['variance_ratio'] = self.run_variance_ratio_test(market_data['log_return'])
        results['time_varying'] = self.analyze_time_varying_efficiency(market_data['log_return'])
        
        # Calculate an efficiency score
        results['efficiency_score'] = self.calculate_efficiency_score(results)
        
        # Determine efficiency classification
        if results['efficiency_score'] >= 80:
            results['efficiency_class'] = 'Highly Efficient'
        elif results['efficiency_score'] >= 60:
            results['efficiency_class'] = 'Moderately Efficient'
        elif results['efficiency_score'] >= 40:
            results['efficiency_class'] = 'Slightly Inefficient'
        else:
            results['efficiency_class'] = 'Highly Inefficient'
        
        if verbose:
            self.print_efficiency_summary(results)
        
        return results
    
    def calculate_efficiency_score(self, results):
        """Calculate a market efficiency score based on test results with improved parameters"""
        score = 0
        max_points = 0
        
        # 1. Non-stationary price series (random walk)
        if 'adf_price' in results and results['adf_price']:
            max_points += 25
            # Higher p-value means more likely to be non-stationary (random walk)
            p_value = results['adf_price']['p_value']
            if p_value > 0.05:  # Not stationary (efficient)
                score += 25
            elif p_value > 0.01:  # Borderline
                score += 15
        
        # 2. Stationary returns
        if 'adf_return' in results and results['adf_return']:
            max_points += 25
            # Lower p-value means more likely to be stationary
            p_value = results['adf_return']['p_value']
            if p_value < 0.01:  # Strongly stationary (efficient)
                score += 25
            elif p_value < 0.05:  # Moderately stationary
                score += 15
        
        # 3. No autocorrelation in returns - more granular scoring
        if 'autocorrelation' in results and results['autocorrelation']:
            max_points += 25
            sig_lags = results['autocorrelation'].get('significant_lags', [])
            
            if not sig_lags:  # No autocorrelation (efficient)
                score += 25
            elif len(sig_lags) == 1 and 1 in sig_lags:  # Only lag 1 is significant
                score += 15
            elif len(sig_lags) <= 2:  # Limited autocorrelation
                score += 10
        
        # 4. Returns not predictable with AR model - more granular scoring
        if 'ar_model' in results and results['ar_model']:
            max_points += 25
            p_value = results['ar_model']['p_value']
            coefficient = abs(results['ar_model']['coefficient'])
            
            if p_value > 0.05:  # Not significant (efficient)
                score += 25
            elif p_value > 0.01 and coefficient < 0.2:  # Weakly significant with small coefficient
                score += 15
            elif coefficient < 0.1:  # Very small coefficient regardless of significance
                score += 10
        
        # Calculate percentage
        if max_points > 0:
            return (score / max_points) * 100
        else:
            return 0
    
    def print_efficiency_summary(self, results):
        """Print a summary of efficiency test results"""
        print("\n" + "="*60)
        print(f"MARKET EFFICIENCY SUMMARY")
        print("="*60)
        print(f"Market: {results['question']}")
        print(f"Market ID: {results['id']}")
        if 'electionType' in results:
            print(f"Type: {results['electionType']}")
        if 'country' in results:
            print(f"Country: {results['country']}")
        print("-"*60)
        
        print("\nEfficiency Tests:")
        
        # Price test
        if 'adf_price' in results and results['adf_price']:
            is_stationary = results['adf_price']['is_stationary']
            print(f"  Price Series: {'Stationary (Inefficient)' if is_stationary else 'Non-stationary - Random Walk (Efficient)'}")
            print(f"    ADF p-value: {results['adf_price']['p_value']:.4f}")
        
        # Returns test
        if 'adf_return' in results and results['adf_return']:
            is_stationary = results['adf_return']['is_stationary']
            print(f"  Returns: {'Stationary (Efficient)' if is_stationary else 'Non-stationary (Inefficient)'}")
            print(f"    ADF p-value: {results['adf_return']['p_value']:.4f}")
        
        # Autocorrelation
        if 'autocorrelation' in results and results['autocorrelation']:
            has_autocorr = results['autocorrelation']['has_significant_autocorrelation']
            print(f"  Autocorrelation: {'Present (Inefficient)' if has_autocorr else 'Absent (Efficient)'}")
            if has_autocorr:
                print(f"    Significant lags: {results['autocorrelation']['significant_lags']}")
        
        # AR model
        if 'ar_model' in results and results['ar_model']:
            is_sig = results['ar_model']['is_significant']
            print(f"  AR(1) Model: {'Significant (Inefficient)' if is_sig else 'Not Significant (Efficient)'}")
            print(f"    Coefficient: {results['ar_model']['coefficient']:.6f}, p-value: {results['ar_model']['p_value']:.4f}")
        
        # Variance ratio
        if 'variance_ratio' in results and results['variance_ratio']:
            print("\n  Variance Ratio Test:")
            for period, result in results['variance_ratio'].items():
                print(f"    {period}: {result['variance_ratio']:.4f} - {result['interpretation']}")
        
        # Time-varying efficiency
        if 'time_varying' in results and results['time_varying'] and 'summary' in results['time_varying']:
            print("\n  Time-varying Efficiency:")
            print(f"    Efficiency Change: {results['time_varying']['summary']['efficiency_change']}")
            print(f"    Volatility Change: {results['time_varying']['summary']['volatility_change']*100:.1f}%")
        
        print("\n" + "-"*60)
        print(f"Overall Efficiency Score: {results['efficiency_score']:.1f}/100")
        print(f"Classification: {results['efficiency_class']}")
        print("="*60)
    
    def visualize_market(self, market_id, results=None):
        """Create visualizations for market efficiency analysis"""
        # Get market data
        market_data = self.preprocess_market_data(market_id)
        if market_data is None:
            print(f"Cannot create visualizations for market {market_id}: No data")
            return None
        
        # Get market details
        details = self.get_market_details(market_id)
        market_name = details['question']
        
        # Set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Price Series
        axs[0, 0].plot(market_data.index, market_data['price'], linewidth=2)
        axs[0, 0].set_title(f'Price Series: {market_name}', fontsize=14)
        axs[0, 0].set_xlabel('Date', fontsize=12)
        axs[0, 0].set_ylabel('Price', fontsize=12)
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Log Returns
        axs[0, 1].plot(market_data.index, market_data['log_return'], linewidth=1, color='green')
        axs[0, 1].set_title(f'Log Returns: {market_name}', fontsize=14)
        axs[0, 1].set_xlabel('Date', fontsize=12)
        axs[0, 1].set_ylabel('Log Return', fontsize=12)
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. ACF Plot
        if results and 'autocorrelation' in results and results['autocorrelation']:
            acf_values = results['autocorrelation']['acf_values']
            lags = range(len(acf_values))
            
            axs[1, 0].bar(lags, acf_values, width=0.4)
            
            # Add confidence intervals
            significance = 1.96 / np.sqrt(len(market_data))
            axs[1, 0].axhline(y=0, linestyle='-', color='black')
            axs[1, 0].axhline(y=significance, linestyle='--', color='red', alpha=0.7)
            axs[1, 0].axhline(y=-significance, linestyle='--', color='red', alpha=0.7)
            
            has_autocorr = results['autocorrelation']['has_significant_autocorrelation']
            title = f'Autocorrelation Function: {"Significant" if has_autocorr else "Not Significant"}'
            axs[1, 0].set_title(title, fontsize=14)
            axs[1, 0].set_xlabel('Lag', fontsize=12)
            axs[1, 0].set_ylabel('ACF', fontsize=12)
        else:
            # Calculate ACF directly
            acf_values = acf(market_data['log_return'].dropna(), nlags=10, fft=True)
            lags = range(len(acf_values))
            
            axs[1, 0].bar(lags, acf_values, width=0.4)
            
            # Add confidence intervals
            significance = 1.96 / np.sqrt(len(market_data))
            axs[1, 0].axhline(y=0, linestyle='-', color='black')
            axs[1, 0].axhline(y=significance, linestyle='--', color='red', alpha=0.7)
            axs[1, 0].axhline(y=-significance, linestyle='--', color='red', alpha=0.7)
            
            axs[1, 0].set_title('Autocorrelation Function', fontsize=14)
            axs[1, 0].set_xlabel('Lag', fontsize=12)
            axs[1, 0].set_ylabel('ACF', fontsize=12)
        
        # 4. Price Distribution
        axs[1, 1].hist(market_data['price'], bins=30, alpha=0.7, density=True)
        axs[1, 1].set_title(f'Price Distribution: {market_name}', fontsize=14)
        axs[1, 1].set_xlabel('Price', fontsize=12)
        axs[1, 1].set_ylabel('Density', fontsize=12)
        
        plt.tight_layout()
        
        # Save the figure if results directory exists
        if os.path.exists(self.results_dir):
            safe_name = "".join(c if c.isalnum() else "_" for c in market_name)[:50]
            filename = os.path.join(self.results_dir, f"market_{market_id}_{safe_name}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        
        return fig
    
    def analyze_multiple_markets(self, market_ids, verbose=False):
        """Analyze multiple markets and compare results"""
        results = []
        
        for market_id in tqdm(market_ids, desc="Analyzing markets"):
            result = self.analyze_market(market_id, verbose=verbose)
            if result['analysis_success']:
                results.append(result)
        
        return results
    
    def visualize_efficiency_comparison(self, results):
        """Create visualization comparing efficiency across markets"""
        if not results:
            print("No results to visualize")
            return None
        
        # Extract key metrics
        market_names = [r['question'][:30] + "..." if len(r['question']) > 30 else r['question'] for r in results]
        efficiency_scores = [r['efficiency_score'] for r in results]
        has_autocorr = [r['autocorrelation']['has_significant_autocorrelation'] if 'autocorrelation' in r and r['autocorrelation'] else None for r in results]
        ar_significant = [r['ar_model']['is_significant'] if 'ar_model' in r and r['ar_model'] else None for r in results]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by efficiency score
        sorted_indices = np.argsort(efficiency_scores)
        sorted_names = [market_names[i] for i in sorted_indices]
        sorted_scores = [efficiency_scores[i] for i in sorted_indices]
        
        # Plot efficiency scores
        bars = ax.barh(sorted_names, sorted_scores, color='skyblue')
        
        # Add efficiency classification regions
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=60, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(x=80, color='green', linestyle='--', alpha=0.5)
        
        ax.text(20, len(sorted_names) - 0.5, "Highly Inefficient", rotation=90, va='top', alpha=0.7)
        ax.text(50, len(sorted_names) - 0.5, "Slightly Inefficient", rotation=90, va='top', alpha=0.7)
        ax.text(70, len(sorted_names) - 0.5, "Moderately Efficient", rotation=90, va='top', alpha=0.7)
        ax.text(90, len(sorted_names) - 0.5, "Highly Efficient", rotation=90, va='top', alpha=0.7)
        
        ax.set_title('Market Efficiency Comparison', fontsize=16)
        ax.set_xlabel('Efficiency Score', fontsize=14)
        ax.set_xlim(0, 100)
        
        # Save figure
        plt.tight_layout()
        filename = os.path.join(self.results_dir, "efficiency_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to {filename}")
        
        return fig
    
    def analyze_and_report_results(data_dir='data', results_dir='results/market_efficiency'):
        """
        Analyze and report results from market efficiency tests
        
        Parameters:
        -----------
        data_dir : str
            Directory with the data
        results_dir : str
            Directory to save results
        
        Returns:
        --------
        dict
            Dictionary with summarized results
        """
        # Make sure the results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Load comprehensive results if they exist
        results_file = os.path.join(results_dir, 'comprehensive_results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            # Run the analysis if results don't exist
            results = run_comprehensive_analysis(data_dir, results_dir, market_count=30, verbose=True)
        
        if not results:
            print("No results available")
            return None
        
        # Create a summary of the results
        summary = {
            'weak_form': {},
            'time_varying': {},
            'strong_form': {},
            'insights': []
        }
        
        # 1. Weak-form efficiency summary
        if 'aggregate' in results:
            agg = results['aggregate']
            summary['weak_form'] = {
                'avg_score': agg.get('avg_efficiency_score', None),
                'median_score': agg.get('median_efficiency_score', None),
                'classes': agg.get('efficiency_classes', {})
            }
            
            # Analyze by market type
            if 'weak_form' in results and 'results' in results['weak_form']:
                market_types = {}
                
                for result in results['weak_form']['results']:
                    if 'electionType' in result and result['electionType']:
                        market_type = result['electionType']
                        
                        if market_type not in market_types:
                            market_types[market_type] = []
                        
                        market_types[market_type].append(result['efficiency_score'])
                
                # Calculate average scores by type
                avg_by_type = {}
                for market_type, scores in market_types.items():
                    if len(scores) >= 2:  # Need at least 2 markets
                        avg_by_type[market_type] = np.mean(scores)
                
                summary['weak_form']['by_type'] = avg_by_type
        
        # 2. Time-varying efficiency summary
        if 'time_varying' in results:
            tv = results['time_varying']
            
            if 'efficiency_changes' in tv:
                summary['time_varying']['changes'] = tv['efficiency_changes']
            
            if 'avg_volatility_change' in tv:
                summary['time_varying']['volatility_change'] = tv['avg_volatility_change']
            
            # Analyze trends
            if 'changes' in summary['time_varying']:
                changes = summary['time_varying']['changes']
                total = sum(changes.values())
                
                if total > 0:
                    more_efficient_pct = changes.get('More Efficient', 0) / total * 100
                    less_efficient_pct = changes.get('Less Efficient', 0) / total * 100
                    
                    if more_efficient_pct > 60:
                        summary['insights'].append("Markets tend to become more efficient over their lifecycle")
                    elif less_efficient_pct > 60:
                        summary['insights'].append("Markets tend to become less efficient over their lifecycle")
                    else:
                        summary['insights'].append("No clear trend in efficiency changes over market lifecycle")
        
        # 3. Strong-form efficiency summary
        if 'strong_form' in results:
            sf = results['strong_form']
            
            if 'avg_speed_of_adjustment' in sf:
                summary['strong_form']['speed_of_adjustment'] = sf['avg_speed_of_adjustment']
            
            if 'avg_volatility_ratio' in sf:
                summary['strong_form']['volatility_ratio'] = sf['avg_volatility_ratio']
            
            # Analyze efficiency based on speed of adjustment
            if 'speed_of_adjustment' in summary['strong_form']:
                speed = summary['strong_form']['speed_of_adjustment']
                
                if speed > 0.8:
                    summary['insights'].append("Markets respond very quickly to new information, indicating strong-form efficiency")
                elif speed > 0.5:
                    summary['insights'].append("Markets show moderate speed in incorporating new information")
                else:
                    summary['insights'].append("Markets show relatively slow adjustment to new information")
        
        # 4. Generate overall insights
        if 'weak_form' in summary and 'avg_score' in summary['weak_form']:
            avg_score = summary['weak_form']['avg_score']
            
            if avg_score >= 75:
                summary['insights'].append("Polymarket election markets are generally highly efficient")
            elif avg_score >= 60:
                summary['insights'].append("Polymarket election markets show moderate efficiency")
            elif avg_score >= 45:
                summary['insights'].append("Polymarket election markets show mixed efficiency")
            else:
                summary['insights'].append("Polymarket election markets show significant inefficiencies")
        
        # Additional insights based on market type comparison
        if 'weak_form' in summary and 'by_type' in summary['weak_form']:
            by_type = summary['weak_form']['by_type']
            
            if by_type:
                # Find most and least efficient types
                most_efficient = max(by_type.items(), key=lambda x: x[1])
                least_efficient = min(by_type.items(), key=lambda x: x[1])
                
                if most_efficient[1] - least_efficient[1] > 15:  # Significant difference
                    summary['insights'].append(f"{most_efficient[0]} markets are notably more efficient than {least_efficient[0]} markets")
        
        # Save the summarized results
        with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def generate_report(summary, results_dir='results/market_efficiency'):
        """
        Generate a report for the thesis based on the results
        
        Parameters:
        -----------
        summary : dict
            Dictionary with summarized results
        results_dir : str
            Directory with result files
        
        Returns:
        --------
        str
            Path to the generated report
        """
        report_lines = []
        
        # Create report header
        report_lines.append("# Market Efficiency Analysis of Polymarket Election Markets")
        report_lines.append("\n## Executive Summary")
        
        # Add insights
        if 'insights' in summary and summary['insights']:
            report_lines.append("\nKey findings from the analysis:")
            for insight in summary['insights']:
                report_lines.append(f"- {insight}")
        
        # Add weak-form efficiency results
        report_lines.append("\n## Weak-Form Efficiency")
        
        if 'weak_form' in summary:
            wf = summary['weak_form']
            
            if 'avg_score' in wf:
                report_lines.append(f"\nThe average efficiency score across analyzed markets is {wf['avg_score']:.2f}/100.")
            
            if 'classes' in wf:
                classes = wf['classes']
                total = sum(classes.values())
                
                report_lines.append("\nMarkets were classified as follows:")
                for cls, count in classes.items():
                    report_lines.append(f"- {cls}: {count} markets ({count/total*100:.1f}%)")
            
            if 'by_type' in wf and wf['by_type']:
                report_lines.append("\n### Efficiency by Market Type")
                report_lines.append("\nAverage efficiency scores by market type:")
                
                # Sort by efficiency score
                sorted_types = sorted(wf['by_type'].items(), key=lambda x: x[1], reverse=True)
                
                for market_type, score in sorted_types:
                    report_lines.append(f"- {market_type}: {score:.2f}/100")
                
                # Add figure reference
                report_lines.append("\n![Efficiency by Market Type](efficiency_by_market_type.png)")
        
        # Add time-varying efficiency results
        report_lines.append("\n## Time-Varying Efficiency")
        
        if 'time_varying' in summary:
            tv = summary['time_varying']
            
            if 'changes' in tv:
                changes = tv['changes']
                total = sum(changes.values())
                
                report_lines.append("\nEfficiency changes over market lifecycle:")
                for change, count in changes.items():
                    report_lines.append(f"- {change}: {count} markets ({count/total*100:.1f}%)")
                
                # Add figure reference
                report_lines.append("\n![Time-varying Efficiency Changes](time_varying_efficiency_changes.png)")
            
            if 'volatility_change' in tv:
                vol_change = tv['volatility_change'] * 100
                direction = "increased" if vol_change > 0 else "decreased"
                report_lines.append(f"\nReturn volatility {direction} by an average of {abs(vol_change):.1f}% from early to late market periods.")
        
        # Add strong-form efficiency results
        report_lines.append("\n## Strong-Form Efficiency")
        
        if 'strong_form' in summary:
            sf = summary['strong_form']
            
            if 'speed_of_adjustment' in sf:
                speed = sf['speed_of_adjustment']
                report_lines.append(f"\nThe average speed of price adjustment to significant events is {speed:.4f}, " + 
                                f"suggesting that {'most' if speed > 0.7 else 'some' if speed > 0.4 else 'limited'} " + 
                                f"price adjustment happens within the first hour after an event.")
            
            if 'volatility_ratio' in sf:
                vol_ratio = sf['volatility_ratio']
                report_lines.append(f"\nThe ratio of post-event to pre-event volatility is {vol_ratio:.2f}, " + 
                                f"indicating {'significantly higher' if vol_ratio > 1.5 else 'higher' if vol_ratio > 1.1 else 'similar' if vol_ratio > 0.9 else 'lower'} " + 
                                f"volatility after significant events.")
            
            # Add figure reference
            report_lines.append("\n![Average Event Response](avg_event_response.png)")
        
        # Add conclusion
        report_lines.append("\n## Conclusion")
        
        # Craft conclusion based on results
        if 'weak_form' in summary and 'avg_score' in summary['weak_form']:
            avg_score = summary['weak_form']['avg_score']
            
            if avg_score >= 75:
                report_lines.append("\nPolymarket election markets exhibit high levels of efficiency, with most markets following random walk patterns consistent with the Efficient Market Hypothesis. These findings suggest that these prediction markets effectively aggregate information and provide reliable forecasts of electoral outcomes.")
            elif avg_score >= 60:
                report_lines.append("\nPolymarket election markets demonstrate moderate efficiency. While many markets follow patterns consistent with the Efficient Market Hypothesis, there is evidence of some predictability in price movements. This suggests opportunities for sophisticated traders to potentially exploit market inefficiencies while still providing reasonably reliable forecasts.")
            elif avg_score >= 45:
                report_lines.append("\nPolymarket election markets show mixed efficiency with substantial variation across market types. The presence of predictable patterns in many markets suggests significant opportunities for informed traders, while also raising questions about the reliability of these markets as pure forecasting tools.")
            else:
                report_lines.append("\nPolymarket election markets exhibit notable inefficiencies, with substantial evidence of predictable patterns across most markets. These findings challenge the application of the Efficient Market Hypothesis to these prediction markets and suggest caution when interpreting market prices as probability forecasts.")
        
        # Additional conclusion points
        if 'time_varying' in summary and 'changes' in summary['time_varying']:
            changes = summary['time_varying']['changes']
            total = sum(changes.values())
            
            if total > 0:
                more_efficient_pct = changes.get('More Efficient', 0) / total * 100
                less_efficient_pct = changes.get('Less Efficient', 0) / total * 100
                
                if more_efficient_pct > 60:
                    report_lines.append("\nMarkets tend to become more efficient over their lifecycle, suggesting that information aggregation improves as more traders participate and more information becomes available.")
                elif less_efficient_pct > 60:
                    report_lines.append("\nMarkets tend to become less efficient over their lifecycle, which may indicate increasing influence of non-information-based trading or herding behavior as elections approach.")
        
        # Add implications section
        report_lines.append("\n## Implications")
        report_lines.append("\nThese findings have several implications for different stakeholders:")
        
        report_lines.append("\n### For Traders")
        report_lines.append("- Trading strategies should be tailored to market type, as efficiency varies significantly across election categories")
        report_lines.append("- Markets may present exploitable patterns despite overall moderate efficiency")
        report_lines.append("- Attention to the timing of trades is important as efficiency changes throughout market lifecycle")
        
        report_lines.append("\n### For Platform Operators")
        report_lines.append("- Market design and incentives can be optimized to improve efficiency where lacking")
        report_lines.append("- Promoting liquidity in certain market types may enhance information aggregation")
        report_lines.append("- Monitoring for manipulative trading is particularly important in less efficient markets")
        
        report_lines.append("\n### For Users of Forecasts")
        report_lines.append("- Prediction market prices should be interpreted with an understanding of varying efficiency")
        report_lines.append("- Complementing market forecasts with other methods may provide more robust predictions")
        report_lines.append("- Special attention should be paid to how markets respond to significant news events")
        
        # Save the report
        report_path = os.path.join(results_dir, "market_efficiency_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        print(f"Report generated at: {report_path}")
        return report_path