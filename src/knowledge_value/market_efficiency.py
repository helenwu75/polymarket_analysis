import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.ar_model import AutoReg
import warnings

class MarketEfficiencyAnalyzer:
    """
    A class for analyzing weak-form market efficiency in prediction markets.
    
    This analyzer implements the key statistical tests for weak-form efficiency:
    1. Random walk test (Augmented Dickey-Fuller)
    2. Autocorrelation test
    3. Runs test for randomness
    4. Autoregressive model test
    """
    
    def __init__(self, results_dir='results/market_efficiency'):
        """Initialize the analyzer with a results directory"""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    # Update the preprocess_market_data method in src/knowledge_value/market_efficiency.py

    def preprocess_market_data(self, trade_data, resample_freq='1min'):
        """
        Preprocess market trade data for efficiency tests
        
        Parameters:
        -----------
        trade_data : pd.DataFrame
            DataFrame with trade data including 'timestamp' and 'price' columns
        resample_freq : str
            Frequency for resampling data (e.g., '1min', '5min', '1h')
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data with regular time intervals
        """
        if trade_data is None or len(trade_data) < 30:
            print("Insufficient trade data for analysis")
            return None
        # Add this check in preprocess_market_data
        if 'token_type' in trade_data.columns and len(trade_data['token_type'].unique()) > 1:
            print("Warning: Mixed token types detected in trade data. Results may be inconsistent.")
            
        try:
            # Ensure timestamp is datetime
            if 'timestamp' in trade_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(trade_data['timestamp']):
                    trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
                
                # Set timestamp as index
                trade_data = trade_data.set_index('timestamp')
            
            # Extract price series
            if 'price' in trade_data.columns:
                # Convert price to numeric type first to ensure calculations work
                trade_data['price'] = pd.to_numeric(trade_data['price'], errors='coerce')
                
                # Drop rows with NaN prices
                trade_data = trade_data.dropna(subset=['price'])
                
                # Resample to regular intervals
                price_series = trade_data['price'].resample(resample_freq).last().ffill()  # Using ffill() instead of fillna(method='ffill')
                
                # Calculate log returns
                log_returns = np.log(price_series / price_series.shift(1)).dropna()
                
                # Create output DataFrame
                result = pd.DataFrame({
                    'price': price_series,
                    'log_return': log_returns
                })
                
                return result
            else:
                print("No price data available")
                return None
        except Exception as e:
            print(f"Error preprocessing market data: {e}")
            return None
    
    # Add this to the market_efficiency.py file

    def generate_detailed_report(self, result, processed_data, output_path=None):
        """
        Generate a detailed markdown report of the market efficiency analysis
        
        Parameters:
        -----------
        result : dict
            Dictionary with analysis results
        processed_data : pd.DataFrame
            Preprocessed market data
        output_path : str, optional
            Path to save the report
            
        Returns:
        --------
        str
            Markdown formatted report
        """
        if not result['analysis_success']:
            return f"# Market Analysis Failed\n\nReason: {result.get('reason', 'Unknown')}"
        
        market_name = result.get('market_name', 'Unknown Market')
        market_id = result.get('market_id', 'Unknown ID')
        
        # Create report header
        report = [
            f"# Market Efficiency Analysis: {market_name}",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Market ID: {market_id}",
            "",
            "## Summary",
            f"**Efficiency Score**: {result['efficiency_score']:.2f}/100",
            f"**Classification**: {result['efficiency_class']}",
            "",
            "## Data Overview",
            f"- **Time Period**: {processed_data.index.min().strftime('%Y-%m-%d')} to {processed_data.index.max().strftime('%Y-%m-%d')}",
            f"- **Number of Data Points**: {len(processed_data):,}",
            f"- **Price Range**: {processed_data['price'].min():.4f} to {processed_data['price'].max():.4f}",
            f"- **Average Price**: {processed_data['price'].mean():.4f}",
            f"- **Price Volatility**: {processed_data['price'].std():.4f}",
            f"- **Return Volatility**: {processed_data['log_return'].std():.4f}",
            "",
            "## Efficiency Tests",
            ""
        ]
        
        # Add random walk test results
        if 'adf_price' in result and result['adf_price']:
            is_random_walk = not result['adf_price']['is_stationary']
            report.extend([
                "### 1. Random Walk Test (Augmented Dickey-Fuller)",
                f"- **Result**: {'PASS' if is_random_walk else 'FAIL'}",
                f"- **Test Statistic**: {result['adf_price']['statistic']:.4f}",
                f"- **P-value**: {result['adf_price']['p_value']:.6f}",
                f"- **Critical Values**:",
                f"  - 1%: {result['adf_price']['critical_values']['1%']:.4f}",
                f"  - 5%: {result['adf_price']['critical_values']['5%']:.4f}",
                f"  - 10%: {result['adf_price']['critical_values']['10%']:.4f}",
                f"- **Interpretation**: {'Price series follows a random walk' if is_random_walk else 'Price series does not follow a random walk (inefficient)'}",
                ""
            ])
        
        # Add return stationarity test results
        if 'adf_return' in result and result['adf_return']:
            is_stationary = result['adf_return']['is_stationary']
            report.extend([
                "### 2. Return Stationarity Test (Augmented Dickey-Fuller)",
                f"- **Result**: {'PASS' if is_stationary else 'FAIL'}",
                f"- **Test Statistic**: {result['adf_return']['statistic']:.4f}",
                f"- **P-value**: {result['adf_return']['p_value']:.6f}",
                f"- **Critical Values**:",
                f"  - 1%: {result['adf_return']['critical_values']['1%']:.4f}",
                f"  - 5%: {result['adf_return']['critical_values']['5%']:.4f}",
                f"  - 10%: {result['adf_return']['critical_values']['10%']:.4f}",
                f"- **Interpretation**: {'Returns are stationary (efficient)' if is_stationary else 'Returns are not stationary (inefficient)'}",
                ""
            ])
        
        # Add autocorrelation test results
        if 'autocorrelation' in result and result['autocorrelation']:
            no_autocorr = not result['autocorrelation']['has_significant_autocorrelation']
            significant_lags = result['autocorrelation']['significant_lags']
            report.extend([
                "### 3. Autocorrelation Test",
                f"- **Result**: {'PASS' if no_autocorr else 'FAIL'}",
                f"- **Significant Lags**: {significant_lags if significant_lags else 'None'}",
                f"- **Interpretation**: {'No significant autocorrelation (efficient)' if no_autocorr else 'Significant autocorrelation detected (inefficient)'}",
                ""
            ])
        
        # Add runs test results
        if 'runs_test' in result and result['runs_test']:
            is_random = result['runs_test']['is_random']
            report.extend([
                "### 4. Runs Test for Randomness",
                f"- **Result**: {'PASS' if is_random else 'FAIL'}",
                f"- **Runs**: {result['runs_test']['runs']:.0f}",
                f"- **Expected Runs**: {result['runs_test']['expected_runs']:.2f}",
                f"- **Z-statistic**: {result['runs_test']['z_statistic']:.4f}",
                f"- **P-value**: {result['runs_test']['p_value']:.6f}",
                f"- **Interpretation**: {'Returns sequence is random (efficient)' if is_random else 'Returns sequence is not random (inefficient)'}",
                ""
            ])
        
        # Add AR model test results
        if 'ar_model' in result and result['ar_model']:
            not_predictable = not result['ar_model']['is_significant']
            report.extend([
                "### 5. Autoregressive Model Test",
                f"- **Result**: {'PASS' if not_predictable else 'FAIL'}",
                f"- **AR(1) Coefficient**: {result['ar_model']['coefficient']:.6f}",
                f"- **P-value**: {result['ar_model']['p_value']:.6f}",
                f"- **Interpretation**: {'Returns are not predictable using AR model (efficient)' if not_predictable else 'Returns are predictable using AR model (inefficient)'}",
                ""
            ])
        
        # Add conclusion
        report.extend([
            "## Conclusion",
            f"This market is classified as **{result['efficiency_class']}** with an efficiency score of **{result['efficiency_score']:.2f}/100**."
        ])
        
        if result['efficiency_score'] < 40:
            report.append("The market exhibits significant inefficiencies, with strong evidence against the random walk hypothesis. The price movements show predictable patterns that could potentially be exploited by traders.")
        elif result['efficiency_score'] < 60:
            report.append("The market exhibits some inefficiencies, with mixed evidence regarding the random walk hypothesis. While some tests indicate efficiency, others suggest the presence of predictable patterns.")
        elif result['efficiency_score'] < 80:
            report.append("The market is moderately efficient, with most tests supporting the random walk hypothesis. There are limited opportunities for exploiting predictable patterns.")
        else:
            report.append("The market is highly efficient, strongly supporting the random walk hypothesis. Price movements appear to be unpredictable, consistent with the efficient market hypothesis.")
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write('\n'.join(report))
            print(f"Detailed report saved to: {output_path}")
        
        return '\n'.join(report)

    def run_adf_test(self, series):
        """
        Run Augmented Dickey-Fuller test for stationarity
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
            
        Returns:
        --------
        dict
            Dictionary with test results
        """
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
        """
        Test for autocorrelation in returns
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        lags : int
            Number of lags to test
            
        Returns:
        --------
        dict
            Dictionary with test results
        """
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
    
    def run_runs_test(self, series):
        """
        Run runs test for randomness
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
            
        Returns:
        --------
        dict
            Dictionary with test results
        """
        try:
            # Calculate signs of returns (positive or negative)
            signs = np.sign(series.dropna())
            
            # Count runs
            runs = len([i for i in range(1, len(signs)) if signs.iloc[i] != signs.iloc[i-1]]) + 1
            
            # Count positive and negative returns
            pos = sum(signs > 0)
            neg = sum(signs < 0)
            
            # Calculate expected runs and standard deviation
            expected_runs = ((2 * pos * neg) / (pos + neg)) + 1
            std_runs = np.sqrt((2 * pos * neg * (2 * pos * neg - pos - neg)) / 
                              ((pos + neg)**2 * (pos + neg - 1)))
            
            # Calculate z-statistic
            z = (runs - expected_runs) / std_runs
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            return {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_statistic': z,
                'p_value': p_value,
                'is_random': p_value >= 0.05
            }
        except Exception as e:
            print(f"Error in runs test: {e}")
            return None
    
    def fit_ar_model(self, series, lags=1):
        """
        Fit autoregressive model to test predictability
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        lags : int
            Number of lags in the AR model
            
        Returns:
        --------
        dict
            Dictionary with test results
        """
        try:
            model = AutoReg(series.dropna(), lags=lags)
            model_fit = model.fit()
            
            # Extract coefficient and p-value
            coef = model_fit.params.iloc[1] if len(model_fit.params) > 1 else 0
            p_value = model_fit.pvalues.iloc[1] if len(model_fit.pvalues) > 1 else 1
            
            return {
                'coefficient': coef,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        except Exception as e:
            print(f"Error in AR model: {e}")
            return None
    
    def analyze_market(self, market_data, market_id=None, market_name=None):
        """
        Run comprehensive efficiency analysis on a market
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Preprocessed market data with price and log_return columns
        market_id : str or int, optional
            ID of the market (for reporting)
        market_name : str, optional
            Name of the market (for reporting)
            
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        if market_data is None or len(market_data) < 30:
            print(f"Insufficient data for market {market_id or ''}")
            return {
                'market_id': market_id,
                'market_name': market_name,
                'analysis_success': False,
                'reason': 'Insufficient data'
            }
        
        results = {
            'market_id': market_id,
            'market_name': market_name,
            'analysis_success': True,
            'data_points': len(market_data)
        }
        
        # Run weak-form efficiency tests
        results['adf_price'] = self.run_adf_test(market_data['price'])
        results['adf_return'] = self.run_adf_test(market_data['log_return'])
        results['autocorrelation'] = self.run_autocorrelation_test(market_data['log_return'])
        results['runs_test'] = self.run_runs_test(market_data['log_return'])
        results['ar_model'] = self.fit_ar_model(market_data['log_return'])
        
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
        
        return results
    
    def calculate_efficiency_score(self, results):
        """
        Calculate a market efficiency score based on test results
        
        Parameters:
        -----------
        results : dict
            Dictionary with test results
            
        Returns:
        --------
        float
            Efficiency score (0-100)
        """
        score = 0
        max_points = 0
        
        # 1. Non-stationary price series (random walk)
        if 'adf_price' in results and results['adf_price']:
            max_points += 25
            p_value = results['adf_price']['p_value']
            if p_value > 0.05:  # Not stationary (efficient)
                score += 25
            elif p_value > 0.01:  # Borderline
                score += 15
        
        # 2. Stationary returns
        if 'adf_return' in results and results['adf_return']:
            max_points += 25
            p_value = results['adf_return']['p_value']
            if p_value < 0.01:  # Strongly stationary (efficient)
                score += 25
            elif p_value < 0.05:  # Moderately stationary
                score += 15
        
        # 3. No autocorrelation in returns
        if 'autocorrelation' in results and results['autocorrelation']:
            max_points += 20
            sig_lags = results['autocorrelation'].get('significant_lags', [])
            
            if not sig_lags:  # No autocorrelation (efficient)
                score += 20
            elif len(sig_lags) == 1 and 1 in sig_lags:  # Only lag 1 is significant
                score += 10
            elif len(sig_lags) <= 2:  # Limited autocorrelation
                score += 5
        
        # 4. Randomness (runs test)
        if 'runs_test' in results and results['runs_test']:
            max_points += 15
            is_random = results['runs_test'].get('is_random', False)
            if is_random:  # Random (efficient)
                score += 15
        
        # 5. Returns not predictable with AR model
        if 'ar_model' in results and results['ar_model']:
            max_points += 15
            is_significant = results['ar_model'].get('is_significant', False)
            if not is_significant:  # Not predictable (efficient)
                score += 15
        
        # Calculate percentage
        if max_points > 0:
            return (score / max_points) * 100
        else:
            return 0
    def visualize_market(market_data, results, market_name=None, save_path=None, academic_style=True):
        """
        Create enhanced academic-style visualizations for market efficiency analysis
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Preprocessed market data
        results : dict
            Dictionary with analysis results
        market_name : str, optional
            Name of the market (for titles)
        save_path : str, optional
            Path to save the visualization
        academic_style : bool
            Whether to use academic styling (serif fonts, etc.)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with visualizations
        """
        if market_data is None:
            return None
        
        # Set academic style if requested
        if academic_style:
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'Computer Modern Roman'],
                'font.size': 11,
                'axes.titlesize': 12,
                'axes.titleweight': 'bold',
                'axes.labelsize': 11,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 9,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
        
        # Set up the figure with more precise layout
        fig = plt.figure(figsize=(9, 10))
        
        # Define grid layout with GridSpec for better control
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                            hspace=0.3, wspace=0.2)
        
        # Create axes for each subplot
        ax1 = fig.add_subplot(gs[0, 0])  # Price series
        ax2 = fig.add_subplot(gs[0, 1])  # Log returns
        ax3 = fig.add_subplot(gs[1, :])  # Autocorrelation
        ax4 = fig.add_subplot(gs[2, :])  # Summary box (will be invisible)
        
        # 1. Price Series - Enhanced
        market_data['price'].plot(ax=ax1, linewidth=1.2, color='navy')
        title = f'Price Series' 
        if market_name:
            ax1.set_title(market_name, fontsize=12, fontweight='bold')
        else:
            ax1.set_title(title, fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add min/max/mean lines for price
        price_min = market_data['price'].min()
        price_max = market_data['price'].max()
        price_mean = market_data['price'].mean()
        
        ax1.axhline(y=price_mean, color='darkred', linestyle='-', linewidth=1, alpha=0.6, 
                    label=f'Mean: {price_mean:.4f}')
        
        # Improve x-axis formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Add legend
        ax1.legend(frameon=True, framealpha=0.7, fontsize=9)
        
        # 2. Log Returns - Enhanced
        market_data['log_return'].plot(ax=ax2, linewidth=0.8, color='darkgreen', alpha=0.8)
        ax2.set_title('Log Returns', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Log Return', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Improve x-axis formatting
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.tick_params(axis='x', rotation=45)
        
        # Add volatility annotation
        volatility = market_data['log_return'].std()
        ax2.annotate(f'Volatility: {volatility:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", alpha=0.7),
                    fontsize=9)
        
        # 3. ACF Plot - Enhanced
        if 'autocorrelation' in results and results['autocorrelation']:
            acf_values = results['autocorrelation']['acf_values']
            lags = range(len(acf_values))
            
            # Plot ACF bars
            bars = ax3.bar(lags, acf_values, width=0.3, color='indigo', alpha=0.8)
            
            # Add confidence intervals with better visibility
            significance = 1.96 / np.sqrt(len(market_data))
            ax3.axhline(y=0, linestyle='-', color='black', linewidth=1)
            ax3.axhline(y=significance, linestyle='--', color='crimson', linewidth=1.2, alpha=0.7,
                    label=f'95% Confidence ({significance:.3f})')
            ax3.axhline(y=-significance, linestyle='--', color='crimson', linewidth=1.2, alpha=0.7)
            
            # Highlight significant lags
            significant_lags = results['autocorrelation'].get('significant_lags', [])
            for lag in significant_lags:
                if lag < len(bars):
                    bars[lag].set_color('red')
                    bars[lag].set_alpha(1.0)
            
            # Better title with test results
            has_autocorr = results['autocorrelation']['has_significant_autocorrelation']
            title = f'Autocorrelation Function: {"Significant" if has_autocorr else "Not Significant"}'
            ax3.set_title(title, fontsize=12, fontweight='bold')
            ax3.set_xlabel('Lag', fontsize=11)
            ax3.set_ylabel('ACF', fontsize=11)
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
            ax3.legend(loc='upper right')
        
        # 4. Summary of efficiency tests in a professional box
        ax4.axis('off')
        
        # Create text for summary with improved formatting
        summary_text = []
        summary_text.append(f"Market: {market_name or 'Unknown'}")
        summary_text.append(f"Efficiency Score: {results['efficiency_score']:.1f}/100")
        summary_text.append(f"Classification: {results['efficiency_class']}")
        summary_text.append("\nTest Results:")
        
        # Use check marks and cross marks for clarity
        if 'adf_price' in results and results['adf_price']:
            is_random_walk = not results['adf_price']['is_stationary']
            summary_text.append(f"• Random Walk Test: {'✓' if is_random_walk else '✗'} (p={results['adf_price']['p_value']:.4f})")
        
        if 'adf_return' in results and results['adf_return']:
            is_stationary = results['adf_return']['is_stationary']
            summary_text.append(f"• Return Stationarity Test: {'✓' if is_stationary else '✗'} (p={results['adf_return']['p_value']:.4f})")
        
        if 'autocorrelation' in results and results['autocorrelation']:
            no_autocorr = not results['autocorrelation']['has_significant_autocorrelation']
            sig_lags = results['autocorrelation'].get('significant_lags', [])
            summary_text.append(f"• No Autocorrelation Test: {'✓' if no_autocorr else '✗'} " + 
                            (f"(lags: {sig_lags})" if sig_lags else ""))
        
        if 'runs_test' in results and results['runs_test']:
            is_random = results['runs_test']['is_random']
            summary_text.append(f"• Runs Test for Randomness: {'✓' if is_random else '✗'} (p={results['runs_test']['p_value']:.4f})")
        
        if 'ar_model' in results and results['ar_model']:
            not_predictable = not results['ar_model']['is_significant']
            summary_text.append(f"• AR Model Test: {'✓' if not_predictable else '✗'} (p={results['ar_model']['p_value']:.4f})")
        
        # Create a more professional-looking text box
        props = dict(boxstyle='round,pad=1', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
        ax4.text(0.5, 0.5, '\n'.join(summary_text), va='center', ha='center', fontsize=11,
                bbox=props, transform=ax4.transAxes)
        
        # Add a figure title with more detail
        plt.suptitle(f"Market Efficiency Analysis: {market_name}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Add footer with data information
        time_range = f"Time Period: {market_data.index.min().strftime('%Y-%m-%d')} to {market_data.index.max().strftime('%Y-%m-%d')}"
        plt.figtext(0.5, 0.01, time_range, ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save the figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_comparison(results_list, save_path=None, academic_style=True):
        """
        Create enhanced academic-style visualization comparing efficiency across markets
        
        Parameters:
        -----------
        results_list : list
            List of dictionaries with market analysis results
        save_path : str, optional
            Path to save the visualization
        academic_style : bool
            Whether to use academic styling (serif fonts, etc.)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with comparison visualization
        """
        if not results_list:
            return None
        
        # Set academic style if requested
        if academic_style:
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'Computer Modern Roman'],
                'font.size': 11,
                'axes.titlesize': 14,
                'axes.titleweight': 'bold',
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
        
        # Extract key metrics
        market_ids = [r.get('market_id', i) for i, r in enumerate(results_list)]
        market_names = [r.get('market_name', f"Market {r.get('market_id', i)}") for i, r in enumerate(results_list)]
        # Truncate long market names
        market_names = [name[:40] + "..." if name and len(name) > 40 else name for name in market_names]
        efficiency_scores = [r.get('efficiency_score', 0) for r in results_list]
        efficiency_classes = [r.get('efficiency_class', 'Unknown') for r in results_list]
        
        # Create figure with increased size for readability
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by efficiency score
        sorted_indices = np.argsort(efficiency_scores)
        sorted_names = [market_names[i] for i in sorted_indices]
        sorted_scores = [efficiency_scores[i] for i in sorted_indices]
        sorted_ids = [market_ids[i] for i in sorted_indices]
        sorted_classes = [efficiency_classes[i] for i in sorted_indices]
        
        # Create color map based on efficiency classes
        color_map = {
            'Highly Inefficient': 'firebrick',
            'Slightly Inefficient': 'darkorange',
            'Moderately Efficient': 'mediumseagreen',
            'Highly Efficient': 'forestgreen'
        }
        bar_colors = [color_map.get(cls, 'gray') for cls in sorted_classes]
        
        # Plot efficiency scores with improved styling
        bars = ax.barh(sorted_names, sorted_scores, color=bar_colors, height=0.6, 
                    edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add efficiency classification regions with more subtle coloring
        ax.axvspan(0, 40, alpha=0.1, color='firebrick', zorder=0)
        ax.axvspan(40, 60, alpha=0.1, color='darkorange', zorder=0)
        ax.axvspan(60, 80, alpha=0.1, color='mediumseagreen', zorder=0)
        ax.axvspan(80, 100, alpha=0.1, color='forestgreen', zorder=0)
        
        # Add vertical lines at thresholds with improved appearance
        ax.axvline(x=40, color='firebrick', linestyle='--', alpha=0.5, linewidth=1.2, zorder=1)
        ax.axvline(x=60, color='darkorange', linestyle='--', alpha=0.5, linewidth=1.2, zorder=1)
        ax.axvline(x=80, color='forestgreen', linestyle='--', alpha=0.5, linewidth=1.2, zorder=1)
        
        # Add classification labels in a more academic style
        ax.text(20, len(sorted_names) + 0.2, "Highly Inefficient", ha='center', 
            fontsize=10, style='italic', color='darkred', zorder=5)
        ax.text(50, len(sorted_names) + 0.2, "Slightly Inefficient", ha='center', 
            fontsize=10, style='italic', color='darkred', zorder=5)
        ax.text(70, len(sorted_names) + 0.2, "Moderately Efficient", ha='center', 
            fontsize=10, style='italic', color='darkgreen', zorder=5)
        ax.text(90, len(sorted_names) + 0.2, "Highly Efficient", ha='center', 
            fontsize=10, style='italic', color='darkgreen', zorder=5)
        
        # Improve title and labels
        ax.set_title('Market Efficiency Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Efficiency Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Improve y-axis appearance
        ax.tick_params(axis='y', which='major', labelsize=10)
        
        # Add grid for readability
        ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        
        # Add score labels with improved styling
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f"{sorted_scores[i]:.1f}", 
                va='center', ha='left', fontsize=10, fontweight='bold')
        
        # Add summary statistics as an annotation
        avg_score = np.mean(efficiency_scores)
        median_score = np.median(efficiency_scores)
        std_score = np.std(efficiency_scores)
        
        stats_text = (f"Average: {avg_score:.1f}\nMedian: {median_score:.1f}\n"
                    f"Std Dev: {std_score:.1f}\nN = {len(efficiency_scores)}")
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save the figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_summary_report(self, results_list):
        """
        Generate a summary report of market efficiency results
        
        Parameters:
        -----------
        results_list : list
            List of dictionaries with market analysis results
            
        Returns:
        --------
        dict
            Dictionary with summary statistics
        """
        if not results_list:
            return {}
        
        # Extract successful analyses
        successful = [r for r in results_list if r.get('analysis_success', False)]
        total_markets = len(successful)
        
        if total_markets == 0:
            return {'error': 'No successful analyses'}
        
        # Calculate aggregate statistics
        efficiency_scores = [r.get('efficiency_score', 0) for r in successful]
        
        # Count by classification
        classifications = {}
        for r in successful:
            if 'efficiency_class' in r:
                cls = r['efficiency_class']
                classifications[cls] = classifications.get(cls, 0) + 1
        
        # Count test results
        test_results = {
            'random_walk_prices': sum(1 for r in successful if r.get('adf_price', {}).get('is_stationary', True) == False),
            'stationary_returns': sum(1 for r in successful if r.get('adf_return', {}).get('is_stationary', False) == True),
            'no_autocorrelation': sum(1 for r in successful if r.get('autocorrelation', {}).get('has_significant_autocorrelation', True) == False),
            'random_runs': sum(1 for r in successful if r.get('runs_test', {}).get('is_random', False) == True),
            'unpredictable_ar': sum(1 for r in successful if r.get('ar_model', {}).get('is_significant', True) == False)
        }
        
        # Convert to percentages
        test_percentages = {k: v / total_markets * 100 for k, v in test_results.items()}
        
        # Create summary
        summary = {
            'total_markets': total_markets,
            'avg_efficiency_score': np.mean(efficiency_scores),
            'median_efficiency_score': np.median(efficiency_scores),
            'std_efficiency_score': np.std(efficiency_scores),
            'min_efficiency_score': min(efficiency_scores),
            'max_efficiency_score': max(efficiency_scores),
            'classifications': classifications,
            'test_results': test_results,
            'test_percentages': test_percentages
        }
        
        return summary