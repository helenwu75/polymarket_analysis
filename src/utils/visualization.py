#!/usr/bin/env python3
'''
Visualization

This module will contain code for analyzing Polymarket data focusing on visualization.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_prediction_accuracy(df, save_path=None):
    """
    Plot the overall prediction accuracy and distribution of Brier scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart of prediction accuracy
    accuracy = df['prediction_correct'].mean()
    ax1.pie([accuracy, 1-accuracy], labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
    ax1.set_title('Prediction Accuracy', fontsize=14)
    
    # Histogram of Brier scores
    sns.histplot(df['brier_score'], kde=True, ax=ax2)
    ax2.set_title('Distribution of Brier Scores', fontsize=14)
    ax2.set_xlabel('Brier Score')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_by_category(df, category_col, top_n=10, save_path=None):
    """
    Plot prediction accuracy by category (election type or country)
    """
    # Calculate accuracy by category
    accuracy_by_cat = df.groupby(category_col)['prediction_correct'].mean().sort_values(ascending=False)
    counts_by_cat = df.groupby(category_col).size()
    
    # Filter to include only categories with at least 5 markets
    valid_cats = counts_by_cat[counts_by_cat >= 5].index
    accuracy_by_cat = accuracy_by_cat[accuracy_by_cat.index.isin(valid_cats)]
    
    # Take top N categories
    if len(accuracy_by_cat) > top_n:
        accuracy_by_cat = accuracy_by_cat.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(accuracy_by_cat.index, accuracy_by_cat.values * 100, color='#2196F3')
    
    # Add count labels
    for i, bar in enumerate(bars):
        cat = accuracy_by_cat.index[i]
        count = counts_by_cat[cat]
        plt.text(i, bar.get_height() + 1, f'n={count}', ha='center')
    
    plt.title(f'Prediction Accuracy by {category_col}', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)  # Set y-axis to percentage
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_correlation_matrix(df, features=None, save_path=None):
    """
    Plot correlation matrix for selected features
    """
    if features is None:
        # Default set of interesting features
        features = [
            'brier_score', 'price_range', 'price_volatility', 'final_week_momentum',
            'price_fluctuations', 'volumeNum', 'unique_traders_count', 
            'trader_to_trade_ratio', 'two_way_traders_ratio', 'trader_concentration',
            'buy_sell_ratio', 'late_stage_participation', 'market_duration_days'
        ]
    
    # Calculate correlation matrix
    corr = df[features].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=True, fmt='.2f', square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr