import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import os

def visualize_efficiency_distribution(efficiency_scores, classification_thresholds=(40, 60, 80), 
                                     save_path=None, academic_style=True):
    """
    Enhanced academic plot of market efficiency score distribution
    
    Parameters:
    -----------
    efficiency_scores : list or array
        List of efficiency scores
    classification_thresholds : tuple
        Thresholds for efficiency classification
    save_path : str, optional
        Path to save figure
    academic_style : bool
        Whether to use academic styling
    """
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
            'savefig.dpi': 300
        })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use seaborn for better histogram aesthetics
    sns.histplot(efficiency_scores, bins=min(10, len(efficiency_scores)), kde=True, 
                 color='slateblue', alpha=0.7, ax=ax, edgecolor='black', linewidth=1.2)
    
    # Add descriptive statistics
    mean_score = np.mean(efficiency_scores)
    median_score = np.median(efficiency_scores)
    
    # Add mean and median lines with improved styling
    ax.axvline(x=mean_score, color='crimson', linestyle='-', linewidth=1.5, 
              label=f"Mean: {mean_score:.2f}")
    ax.axvline(x=median_score, color='darkgreen', linestyle='--', linewidth=1.5, 
              label=f"Median: {median_score:.2f}")
    
    # Add classification regions with subtle coloring
    ax.axvspan(0, classification_thresholds[0], alpha=0.1, color='firebrick', zorder=0)
    ax.axvspan(classification_thresholds[0], classification_thresholds[1], alpha=0.1, color='darkorange', zorder=0)
    ax.axvspan(classification_thresholds[1], classification_thresholds[2], alpha=0.1, color='mediumseagreen', zorder=0)
    ax.axvspan(classification_thresholds[2], 100, alpha=0.1, color='forestgreen', zorder=0)
    
    # Add vertical threshold lines
    for threshold in classification_thresholds:
        ax.axvline(x=threshold, color='orange', linestyle=':', alpha=0.8, linewidth=1.2)
    
    # Add classification labels
    ymax = ax.get_ylim()[1]
    ax.text(20, ymax * 0.9, "Highly\nInefficient", ha='center', va='top', 
           fontsize=9, style='italic', rotation=90, alpha=0.8)
    ax.text(50, ymax * 0.9, "Slightly\nInefficient", ha='center', va='top', 
           fontsize=9, style='italic', rotation=90, alpha=0.8)
    ax.text(70, ymax * 0.9, "Moderately\nEfficient", ha='center', va='top', 
           fontsize=9, style='italic', rotation=90, alpha=0.8)
    ax.text(90, ymax * 0.9, "Highly\nEfficient", ha='center', va='top', 
           fontsize=9, style='italic', rotation=90, alpha=0.8)
    
    # Improve title and labels
    ax.set_title("Distribution of Market Efficiency Scores", fontsize=14, fontweight='bold')
    ax.set_xlabel("Efficiency Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    # Add a more detailed legend
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Add a summary statistics annotation
    stats_text = (f"n = {len(efficiency_scores)}\n"
                 f"Mean = {mean_score:.2f}\n"
                 f"Median = {median_score:.2f}\n"
                 f"Std Dev = {np.std(efficiency_scores):.2f}\n"
                 f"Min = {min(efficiency_scores):.2f}\n"
                 f"Max = {max(efficiency_scores):.2f}")
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Set x-axis limits for consistency
    ax.set_xlim(0, 100)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_test_results(test_percentages, save_path=None, academic_style=True):
    """
    Create enhanced academic visualization of test results
    
    Parameters:
    -----------
    test_percentages : dict
        Dictionary with test results percentages
    save_path : str, optional
        Path to save the visualization
    academic_style : bool
        Whether to use academic styling
    """
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
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract test names and percentages with improved labels
    test_labels = {
        'random_walk_prices': 'Random Walk\nPrices',
        'stationary_returns': 'Stationary\nReturns',
        'no_autocorrelation': 'No\nAutocorrelation',
        'random_runs': 'Random\nRuns',
        'unpredictable_ar': 'Unpredictable\nAR Model'
    }
    
    test_names = [test_labels.get(test, test.replace('_', ' ').title()) 
                 for test in test_percentages.keys()]
    percentages = list(test_percentages.values())
    
    # Create the bar chart with improved styling
    bars = ax.bar(test_names, percentages, color='skyblue', width=0.6,
                 edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Add percentage labels with improved styling
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f"{percentages[i]:.1f}%", 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Improve title and labels
    ax.set_title("Percentage of Markets Passing Each Efficiency Test", fontsize=14, fontweight='bold')
    ax.set_ylabel("Percentage", fontsize=12)
    
    # Enhance tick labels
    plt.xticks(rotation=0, ha='center')
    
    # Set y-axis limits
    ax.set_ylim(0, 100 + 5)  # Add space for labels
    
    # Add a reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, 50, "50%", va='center', ha='right', 
           fontsize=9, style='italic', color='gray')
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add interpretation note
    note_text = "Note: Higher percentages indicate better market efficiency across the analyzed markets."
    plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_classification_pie(classifications, save_path=None, academic_style=True):
    """
    Create enhanced academic pie chart of market efficiency classes
    
    Parameters:
    -----------
    classifications : dict
        Dictionary with classification counts
    save_path : str, optional
        Path to save visualization
    academic_style : bool
        Whether to use academic styling
    """
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
            'savefig.dpi': 300
        })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract labels and sizes
    labels = list(classifications.keys())
    sizes = list(classifications.values())
    total = sum(sizes)
    
    # Create color mapping with consistent academic colors
    colors = {
        'Highly Inefficient': 'firebrick',
        'Slightly Inefficient': 'darkorange',
        'Moderately Efficient': 'mediumseagreen',
        'Highly Efficient': 'forestgreen'
    }
    
    # Get colors for each label
    pie_colors = [colors.get(label, 'gray') for label in labels]
    
    # Create enhanced pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,  # We'll add labels separately for better control
        colors=pie_colors,
        autopct=lambda p: f'{p:.1f}%\n({int(p*total/100)})',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        textprops={'fontsize': 11, 'color': 'white', 'fontweight': 'bold'},
        shadow=False,
        explode=[0.05 if label == 'Highly Inefficient' else 0 for label in labels]  # Explode the most inefficient wedge
    )
    
    # Enhance the appearance of percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Create a custom legend with percentages
    legend_labels = [f"{label} ({size/total*100:.1f}%)" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1))
    
    # Add title with enhanced styling
    ax.set_title('Market Efficiency Classification', fontsize=14, fontweight='bold', pad=20)
    
    # Add annotation with total count
    ax.annotate(f'Total Markets: {total}', xy=(0, -0.1), xycoords='axes fraction',
               ha='center', va='center', fontsize=10)
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_all_visualizations(results, results_dir, academic_style=True):
    """
    Generate and save all visualizations for market efficiency analysis
    
    Parameters:
    -----------
    results : dict
        Dictionary with analysis results
    results_dir : str
        Directory to save results
    academic_style : bool
        Whether to use academic styling
    """
    import os
    
    # Create directories for results
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract relevant results
    results_list = results.get('market_results', [])
    summary = results.get('summary', {})
    
    # 1. Individual market visualizations
    for result in results_list:
        if result.get('analysis_success', False):
            market_id = result.get('market_id', 'unknown')
            market_name = result.get('market_name', f"Market {market_id}")
            
            # Get processed data (you'll need to ensure this is available)
            # This might require adjusting your analysis pipeline
            processed_data = get_processed_data_for_market(market_id)
            
            if processed_data is not None:
                safe_name = ''.join(c if c.isalnum() else '_' for c in market_name[:40])
                viz_path = os.path.join(results_dir, f"market_{market_id}_{safe_name}.png")
                visualize_market(processed_data, result, market_name, viz_path, academic_style=academic_style)
    
    # 2. Market comparison visualization
    if len(results_list) > 1:
        successful_results = [r for r in results_list if r.get('analysis_success', False)]
        comparison_path = os.path.join(results_dir, "market_comparison.png")
        visualize_comparison(successful_results, comparison_path, academic_style=academic_style)
    
    # 3. Efficiency score distribution
    if 'avg_efficiency_score' in summary:
        efficiency_scores = [r.get('efficiency_score', 0) for r in results_list if r.get('analysis_success', False)]
        distribution_path = os.path.join(results_dir, "efficiency_distribution.png")
        visualize_efficiency_distribution(efficiency_scores, save_path=distribution_path, academic_style=academic_style)
    
    # 4. Classification pie chart
    if 'classifications' in summary:
        pie_path = os.path.join(results_dir, "efficiency_classification.png")
        visualize_classification_pie(summary['classifications'], save_path=pie_path, academic_style=academic_style)
    
    # 5. Test results bar chart
    if 'test_percentages' in summary:
        test_path = os.path.join(results_dir, "test_percentages.png")
        visualize_test_results(summary['test_percentages'], save_path=test_path, academic_style=academic_style)

def get_processed_data_for_market(market_id):
    """
    Helper function to get processed data for a market
    
    This is a placeholder - you'll need to implement this based on your data storage approach.
    One option is to modify your analysis pipeline to store the processed data in the result dict.
    """
    # Placeholder implementation
    # You may need to reload and reprocess the data, or ideally store it during initial analysis
    from utils.data_loader import load_trade_data
    from knowledge_value.market_efficiency import MarketEfficiencyAnalyzer
    
    analyzer = MarketEfficiencyAnalyzer()
    trade_data = load_trade_data(market_id, token_type='yes', return_all_tokens=False)
    
    if trade_data is not None and len(trade_data) >= 30:
        return analyzer.preprocess_market_data(trade_data)
    return None