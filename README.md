# Polymarket Analysis Repository

This repository contains code and analysis scripts for examining Polymarket prediction markets, focusing on understanding their role in creating different forms of value: knowledge value, financial value, cultural value, and identity value.

## Repository Structure

```
polymarket_analysis/
├── data/                          # Processed data files
│   ├── cleaned_election_data.csv  # Main dataset with cleaned market data
│   ├── trades/                    # Directory with trade-level data
│   ├── market_tokens.json         # Mapping of markets to tokens
│   └── external/                  # External data (polls, stock market, etc.)
├── src/                           # Source code for analyses
│   ├── knowledge_value/           # Scripts for knowledge value analyses
│   │   ├── price_efficiency.py    # Tests for random walks and market efficiency
│   │   ├── event_study.py         # Event study analysis
│   │   └── forecast_comparison.py # Comparison with alternative forecasting methods
│   ├── financial_value/           # Scripts for financial value analyses
│   │   ├── market_correlation.py  # Correlation with traditional markets
│   │   └── hedging_analysis.py    # Analysis of hedging potential
│   ├── cultural_value/            # Scripts for cultural value analyses
│   │   ├── trader_concentration.py # Trader concentration analysis
│   │   ├── profit_distribution.py  # Distribution of trader profits
│   │   └── whale_impact.py         # Analysis of large trader impact
│   ├── identity_value/            # Scripts for identity value analyses
│   │   └── trader_typology.py     # Classification of trader types
│   └── utils/                     # Utility functions
│       ├── data_loader.py         # Functions to load and process data
│       ├── visualization.py       # Common visualization functions
│       └── statistics.py          # Statistical utility functions
├── notebooks/                     # Jupyter notebooks for exploration and visualization
│   ├── knowledge_value_analysis.ipynb
│   ├── financial_value_analysis.ipynb
│   ├── cultural_value_analysis.ipynb
│   └── identity_value_analysis.ipynb
├── results/                       # Output directory for results and visualizations
│   ├── knowledge_value/
│   ├── financial_value/
│   ├── cultural_value/
│   └── identity_value/
├── case_studies/                  # Detailed case studies of specific markets
│   ├── us_presidential.py
│   ├── french_election.py
│   └── regional_election.py
├── requirements.txt               # Python dependencies
├── run_all.py                     # Script to execute all analyses
└── README.md                      # This file
```

## Analysis Modules

### Knowledge Value Analysis

1. **Price Efficiency Tests** (`src/knowledge_value/price_efficiency.py`):

   - Random walk tests to assess market efficiency
   - Autocorrelation analysis to detect predictable patterns
   - Variance ratio tests to identify market inefficiencies

2. **Event Studies** (`src/knowledge_value/event_study.py`):

   - Analysis of price reactions to major events
   - Price charts and statistical analysis for events in:
     - US Presidential Election
     - French RN/UXD Election
     - Local/Regional Elections

3. **Forecasting Comparison** (`src/knowledge_value/forecast_comparison.py`):
   - Comparative analysis of prediction markets vs polls and expert forecasts
   - Calculate accuracy metrics across different forecasting methods
   - Statistical tests of comparative performance

### Financial Value Analysis

1. **Market Correlation Analysis** (`src/financial_value/market_correlation.py`):

   - Analysis of correlation between prediction markets and traditional financial markets
   - Tests for diversification potential and hedging effectiveness
   - Time-varying correlation analysis

2. **Hedging Analysis** (`src/financial_value/hedging_analysis.py`):
   - Sector-specific hedging potential
   - Portfolio construction with prediction markets
   - Risk-return profile of prediction market portfolios

### Cultural Value Analysis

1. **Trader Concentration Analysis** (`src/cultural_value/trader_concentration.py`):

   - Measures of market concentration
   - Impact of concentration on market efficiency and accuracy
   - Temporal evolution of market concentration

2. **Profit Distribution Analysis** (`src/cultural_value/profit_distribution.py`):

   - Gini coefficient calculations for trader profits
   - Lorenz curve visualizations
   - Comparative analysis across market types

3. **Whale Impact Analysis** (`src/cultural_value/whale_impact.py`):
   - Identification of large traders and their impact
   - Price impact regression analysis
   - Market microstructure effects of large trades

### Identity Value Analysis

1. **Trader Typology** (`src/identity_value/trader_typology.py`):
   - Clustering analysis of trader behaviors
   - Feature engineering for trader classification
   - Visualization and interpretation of trader types

## Case Studies

The `case_studies/` directory contains in-depth analyses of specific markets:

1. **US Presidential Election** (`case_studies/us_presidential.py`):

   - High-volume, high-stakes market analysis
   - Detailed price and volume analysis
   - Trader behavior and concentration metrics

2. **French RN/UXD Election** (`case_studies/french_election.py`):

   - Analysis of a market that failed to correctly predict the outcome
   - Examination of information environment and trader behavior
   - Post-mortem on prediction failure

3. **Regional Election** (`case_studies/regional_election.py`):
   - Analysis of a smaller market with limited information
   - Examination of trader composition and information asymmetries
   - Comparison with larger, more liquid markets

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages are listed in `requirements.txt`

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/polymarket-analysis.git
cd polymarket-analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up data:

- Place the `cleaned_election_data.csv` file in the `data/` directory
- Organize trade-level data in `data/trades/`
- Add market token mappings to `data/market_tokens.json`

Running Analyses

To run individual analyses:

```bash
# Run price efficiency tests
python src/knowledge_value/price_efficiency.py

# Run event studies
python src/knowledge_value/event_study.py

# Run all analyses
python run_all.py
```
