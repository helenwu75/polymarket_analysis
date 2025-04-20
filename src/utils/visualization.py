#!/usr/bin/env python3
'''
Visualization

This module contains improved visualization functions for Polymarket data analysis.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os

def set_plot_style():
    """Set consistent plot style for all visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def plot_prediction_accuracy(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the overall prediction accuracy and distribution of Brier scores
    
    Args:
        df: DataFrame with prediction metrics
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart of prediction accuracy
    accuracy = df['prediction_correct'].mean()
    wedges, texts, autotexts = ax1.pie(
        [accuracy, 1-accuracy], 
        labels=['Correct', 'Incorrect'], 
        autopct='%1.1f%%', 
        colors=['#4CAF50', '#F44336'],
        explode=(0.05, 0),
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12}
    )
    ax1.set_title('Prediction Accuracy', fontsize=16, pad=20)
    
    # Histogram of Brier scores
    sns.histplot(df['brier_score'], kde=True, ax=ax2, bins=30, color='#2196F3')
    ax2.set_title('Distribution of Brier Scores', fontsize=16)
    ax2.set_xlabel('Brier Score (Lower is Better)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    
    # Add vertical line for mean
    mean_brier = df['brier_score'].mean()
    ax2.axvline(x=mean_brier, color='red', linestyle='--', linewidth=2)
    ax2.text(mean_brier * 1.1, 0.95 * ax2.get_ylim()[1], 
             f'Mean: {mean_brier:.4f}', fontsize=12, color='red')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_accuracy_by_category(df: pd.DataFrame, category_col: str, 
                              top_n: int = 10, min_markets: int = 5,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot prediction accuracy by category (election type or country)
    
    Args:
        df: DataFrame with prediction metrics
        category_col: Column containing categories (e.g., 'event_electionType')
        top_n: Number of top categories to display
        min_markets: Minimum number of markets required for a category
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    set_plot_style()
    
    # Calculate accuracy by category
    accuracy_by_cat = df.groupby(category_col)['prediction_correct'].mean().sort_values(ascending=False)
    counts_by_cat = df.groupby(category_col).size()
    
    # Filter to include only categories with at least min_markets
    valid_cats = counts_by_cat[counts_by_cat >= min_markets].index
    accuracy_by_cat = accuracy_by_cat[accuracy_by_cat.index.isin(valid_cats)]
    
    # Take top N categories
    if len(accuracy_by_cat) > top_n:
        accuracy_by_cat = accuracy_by_cat.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars for better readability with category names
    y_pos = np.arange(len(accuracy_by_cat))
    bars = ax.barh(y_pos, accuracy_by_cat.values * 100, 
                 color=sns.color_palette("viridis", len(accuracy_by_cat)))
    
    # Add count labels
    for i, bar in enumerate(bars):
        cat = accuracy_by_cat.index[i]
        count = counts_by_cat[cat]
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'n={count}', ha='left', va='center', fontsize=10)
    
    # Add percentage labels inside the bars
    for i, v in enumerate(accuracy_by_cat.values):
        if v > 0.3:  # Only add text if bar is wide enough
            ax.text(v * 50, i, f'{v*100:.1f}%', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(accuracy_by_cat.index)
    ax.invert_yaxis()  # To have the highest accuracy at the top
    
    ax.set_title(f'Prediction Accuracy by {category_col}', fontsize=16, pad=20)
    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_xlim(0, 105)  # Set x-axis to percentage
    
    # Add grid for readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_feature_correlation_matrix(df: pd.DataFrame, features: Optional[List[str]] = None, 
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix for selected features
    
    Args:
        df: DataFrame with features
        features: List of features to include in correlation matrix
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object and correlation matrix
    """
    set_plot_style()
    
    if features is None:
        # Default set of interesting features
        features = [
            'brier_score', 'price_range', 'price_volatility', 'final_week_momentum',
            'price_fluctuations', 'volumeNum', 'unique_traders_count', 
            'trader_to_trade_ratio', 'two_way_traders_ratio', 'trader_concentration',
            'buy_sell_ratio', 'late_stage_participation', 'market_duration_days'
        ]
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in df.columns]
    
    # Calculate correlation matrix
    corr = df[features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap with improved aesthetics
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
                annot=True, fmt='.2f', square=True, linewidths=.5,
                cbar_kws={"shrink": .8})
    
    # Add labels and title
    ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, corr

def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                     hue_col: Optional[str] = None, marker: bool = True,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series data with improved aesthetics
    
    Args:
        df: DataFrame with time series data
        x_col: Column for x-axis (usually timestamp)
        y_col: Column for y-axis
        title: Plot title
        hue_col: Column for color grouping
        marker: Whether to show markers
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if hue_col:
        for name, group in df.groupby(hue_col):
            ax.plot(group[x_col], group[y_col], 
                   marker='o' if marker else None, linestyle='-', 
                   markersize=5, label=name, alpha=0.8)
        ax.legend(title=hue_col)
    else:
        ax.plot(df[x_col], df[y_col], 
               marker='o' if marker else None, linestyle='-', 
               markersize=5, color='#2196F3', alpha=0.8)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    
    # Format x-axis for datetime if applicable
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        fig.autofmt_xdate()
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multi_feature_comparison(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                                 n_bins: int = 5, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a multi-panel plot showing relationship between multiple features and target
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature columns to compare
        target_col: Target column (e.g., 'brier_score')
        n_bins: Number of bins for each feature
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    set_plot_style()
    
    # Calculate number of rows and columns for subplots
    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        if i < len(axes):
            ax = axes[i]
            
            # Create bins for the feature
            df_clean = df[[feature, target_col]].dropna()
            if len(df_clean) < n_bins:
                ax.text(0.5, 0.5, f"Insufficient data for {feature}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
                
            bins = pd.qcut(df_clean[feature], n_bins, duplicates='drop')
            
            # Calculate mean target value for each bin
            bin_stats = df_clean.groupby(bins)[target_col].agg(['mean', 'count', 'std'])
            bin_stats['bin_center'] = [group.mean() for group in bins.categories]
            
            # Plot the relationship
            ax.bar(np.arange(len(bin_stats)), bin_stats['mean'], 
                  yerr=bin_stats['std'], alpha=0.7, 
                  color=sns.color_palette("viridis", len(bin_stats)))
            
            # Add count labels
            for j, (_, row) in enumerate(bin_stats.iterrows()):
                ax.text(j, row['mean'] + row['std'] + 0.02, 
                       f'n={int(row["count"])}', ha='center', fontsize=8)
            
            # Format x-axis
            ax.set_xticks(np.arange(len(bin_stats)))
            x_labels = [f'{c.left:.2f}-{c.right:.2f}' for c in bins.categories]
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            ax.set_title(f'{feature} vs {target_col}', fontsize=12)
            ax.set_ylabel(target_col, fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Feature Comparison with {target_col}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_lorenz_curve(trader_volumes: pd.DataFrame, gini_coef: float, title: str,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Lorenz curve for trader volume distribution
    
    Args:
        trader_volumes: DataFrame with trader volumes
        gini_coef: Gini coefficient 
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    set_plot_style()
    
    # Sort by volume share for Lorenz curve
    trader_volumes_sorted = trader_volumes.sort_values('volume_share')
    trader_volumes_sorted['lorenz_curve'] = trader_volumes_sorted['volume_share'].cumsum()
    perfect_equality = np.linspace(0, 1, len(trader_volumes_sorted))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Lorenz curve
    ax.plot(np.linspace(0, 1, len(trader_volumes_sorted)), trader_volumes_sorted['lorenz_curve'], 
            label='Lorenz Curve', color='#E91E63', linewidth=3)
    
    # Plot line of perfect equality
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=2)
    
    # Fill area between curves
    ax.fill_between(np.linspace(0, 1, len(trader_volumes_sorted)), 
                   perfect_equality, 
                   trader_volumes_sorted['lorenz_curve'],