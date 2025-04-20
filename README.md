# Polymarket Analysis Repository

This repository contains code and analysis scripts for examining Polymarket prediction markets, focusing on understanding their role in creating different forms of value: knowledge value, financial value, cultural value, and identity value.

## Repository Structure

```
polymarket_analysis/
├── data/                          # Processed data files
│   ├── cleaned_election_data.csv  # Main dataset with market features
│   ├── trades/                    # Directory with trade-level data
│   │   ├── matched_events/        # Parquet files for matched events
│   │   ├── market_id_to_question.json # Maps market IDs to question text
│   │   └── market_tokens.json     # Information about market tokens
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

## Planned Analyses

The repository will include the following analyses:

### Knowledge Value Analysis

- Price efficiency tests (random walk, autocorrelation)
- Event studies for major election markets
- Comparison of market predictions to alternative forecasting methods

### Financial Value Analysis

- Correlation analysis with traditional financial markets
- Hedging potential analysis

### Cultural Value Analysis

- Trader concentration analysis (Gini coefficient)
- Profit distribution analysis
- Whale impact analysis

### Identity Value Analysis

- Trader typology through clustering analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages will be listed in `requirements.txt`

### Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Polymarket data to the appropriate directories

### Running Analyses

- Individual analyses can be run directly from their respective scripts
- The `run_all.py` script will execute all analyses in sequence

## License

[Your chosen license]

## Author

[Your name]
