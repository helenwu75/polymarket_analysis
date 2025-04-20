# Polymarket Data Infrastructure

This document provides a comprehensive explanation of the data infrastructure used in the Polymarket analysis project, including the structure of trade data, CSV files, and how the `data_loader.py` script works to access and process this data.

## Data Structure Overview

The project's data is organized in the following structure:

```
data/
├── cleaned_election_data.csv   # Main dataset with market features
├── data-overview.md            # Overview documentation
├── trades/                     # Directory with trade-level data
│   ├── matched_events/         # Parquet files for matched events
│   ├── trades/                 # Directory containing individual trade files
│   │   └── [token_id].parquet  # Trade data files named by token ID
│   ├── orderbooks/             # Directory containing orderbook data
│   ├── market_id_to_question.json  # Mapping between market IDs and questions
│   └── market_tokens.json      # Mapping between market IDs and token IDs
└── external/                   # Directory for external data (empty)
```

## Main Dataset (`cleaned_election_data.csv`)

This is the primary dataset containing aggregated market features:

- **Size**: 1,048,575 rows and 54 columns
- **Key Target Variables**:
  - `brier_score`: Measures prediction accuracy (lower is better)
  - `log_loss`: Alternative measure of prediction accuracy
  - `prediction_correct`: Binary indicator of whether the prediction was correct

### Key Feature Categories

#### Market Identifiers

- `id`: Numeric ID for the market
- `market_id`: Market identifier
- `yes_token_id`: Token ID for the "Yes" outcome
- `clobTokenIds`: List of all token IDs for the market

#### Market Information

- `question`: The market question text
- `event_electionType`: Type of election (Presidential, Senate, etc.)
- `event_country`: Country of the election
- `startDate`, `endDate`: Market start and end dates
- `market_duration_days`: Duration of the market in days

#### Price Dynamics

- `closing_price`: Final market price
- `price_volatility`: Coefficient of variation of prices in final week
- `price_range`: Difference between highest and lowest price in final week
- `final_week_momentum`: Price change over final week
- `price_fluctuations`: Number of times price crossed threshold
- `pre_election_vwap_48h`: Volume-weighted average price 48h before election

#### Trading Activity

- `volumeNum`: Total trading volume
- `trading_frequency`: Average number of trades per day
- `trading_continuity`: Percentage of days with at least one trade
- `buy_sell_ratio`: Ratio of buy orders to sell orders
- `late_stage_participation`: Proportion of trades in final week
- `volume_acceleration`: Ratio of final week trading to overall average

#### Trader Behavior

- `unique_traders_count`: Number of distinct traders
- `trader_to_trade_ratio`: Average number of trades per trader
- `two_way_traders_ratio`: Proportion of traders who both bought and sold
- `trader_concentration`: Proportion of trades by top traders
- `new_trader_influx`: Proportion of traders who first appeared in final week

## Trade-Level Data Structure

The raw trade data is stored in Parquet files in the `data/trades/trades/` directory. Each file corresponds to a specific token ID and contains individual trade records.

### Trade File Structure

Each Parquet file contains the following columns:

- `id`: Unique trade identifier
- `price`: Trade price
- `side`: Buy or Sell
- `size`: Size of the trade
- `timestamp`: When the trade occurred (Unix timestamp)
- `transactionHash`: Blockchain transaction hash
- `maker_id`: Ethereum address of the maker
- `taker_id`: Ethereum address of the taker
- `token_id`: Token ID being traded

### Market ID to Token ID Mapping

The `market_tokens.json` file maps market IDs to their associated token IDs. This is crucial because trade data files are named using token IDs, not market IDs.

Example structure:

```json
{
  "0xf810652Ca2F32CECF67c71adFB534b98B567F344": [
    "23874437153982785160552190848753406234716605383255209659534817782884226760426",
    "107522587888956657725619810357030133777015548886105324400975858186021620696066"
  ]
}
```

### Market ID to Question Mapping

The `market_id_to_question.json` file maps market IDs (usually Ethereum addresses) to the corresponding market questions.

Example structure:

```json
{
  "0xf810652Ca2F32CECF67c71adFB534b98B567F344": "[Single Market] Will Donald J. Trump win the U.S. 2024 Republican presidential nomination?"
}
```

## Data Loader Implementation

The `data_loader.py` script provides functions to load and process data from the project's data structure. Here's an explanation of the key functions:

### Main Functions

#### `load_main_dataset(filepath='data/cleaned_election_data.csv')`

- Loads the main dataset containing market features
- Returns a pandas DataFrame with all market data
- Handles mixed data types with `low_memory=False`

#### `load_market_question_mapping(filepath='data/trades/market_id_to_question.json')`

- Loads the mapping between market IDs and their questions
- Returns a dictionary with market IDs as keys and questions as values

#### `load_market_tokens(filepath='data/trades/market_tokens.json')`

- Loads the mapping between market IDs and their token IDs
- Returns a dictionary with market IDs as keys and lists of token IDs as values

#### `load_trade_data(market_id, trades_dir='data/trades', return_all_tokens=True)`

- Loads trade-level data for a specific market
- Parameters:
  - `market_id`: Market ID to load data for
  - `trades_dir`: Base directory for trade data
  - `return_all_tokens`: If True, returns trades for all tokens in the market
- Process:
  1. Gets token IDs for the market using `get_token_ids_for_market()`
  2. For each token ID, finds the corresponding Parquet file using `find_token_id_file()`
  3. Loads the Parquet file into a DataFrame
  4. Adds market ID information to the DataFrame
  5. Converts Unix timestamps to datetime objects
  6. Combines data from all tokens if requested
- Returns a pandas DataFrame with trade data or None if not found

### Helper Functions

#### `find_token_id_file(token_id, trades_dir='data/trades/trades')`

- Finds the Parquet file that matches a given token ID
- Handles file naming inconsistencies by checking if filenames start with the token ID

#### `get_token_ids_for_market(market_id, market_tokens_filepath='data/trades/market_tokens.json')`

- Gets the list of token IDs associated with a market ID
- Handles case-insensitivity for Ethereum addresses

#### `get_market_id_from_ethereum_address(ethereum_address, mapping_file='data/trades/market_id_to_question.json')`

- Gets a market ID that corresponds to an Ethereum address
- Handles case-insensitivity for Ethereum addresses

#### `get_sample_market_ids(n=5, mapping_file='data/trades/market_id_to_question.json')`

- Gets a sample of valid market IDs from the mapping file
- Useful for testing or sample analysis

#### `summarize_dataset(df)`

- Prints summary statistics and information about the dataset
- Includes accuracy metrics, missing values, key feature summaries, and market distributions

## Data Challenges and Solutions

### Challenge 1: Market ID vs. Token ID Mismatch

The files in the trades directory are named using token IDs, not market IDs. The data loader bridges this gap using the market_tokens.json file to map between them.

### Challenge 2: Ethereum Addresses vs. Numeric IDs

The market IDs in the mapping file are Ethereum addresses (0x...), while the dataset may use numeric IDs. The data loader handles both formats.

### Challenge 3: Timestamp Conversion

The timestamps in the trade data are Unix epoch seconds stored as strings. The data loader converts them to proper datetime objects with:

```python
df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
```

### Challenge 4: Files with Extra Suffixes

Some Parquet files have timestamps or other information appended to their names. The data loader uses a prefix matching approach to find the right files.

### Challenge 5: Multiple Tokens per Market

A market typically has multiple tokens (e.g., YES and NO). The data loader can load data for all tokens in a market and combine them into a single DataFrame.

## Usage Examples

### Loading the Main Dataset

```python
from utils.data_loader import load_main_dataset
df = load_main_dataset()
print(f"Loaded dataset with shape: {df.shape}")
```

### Loading Trade Data for a Specific Market

```python
from utils.data_loader import load_trade_data, load_market_question_mapping

# Load market mapping
mapping = load_market_question_mapping()

# Get a market ID (e.g., first one in the mapping)
market_id = next(iter(mapping))
question = mapping[market_id]
print(f"Loading trades for market: {question}")

# Load trade data
trades_df = load_trade_data(market_id)
if trades_df is not None:
    print(f"Loaded {len(trades_df)} trades from {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
```

### Getting a Sample of Market IDs for Analysis

```python
from utils.data_loader import get_sample_market_ids, load_market_question_mapping

# Get mapping and sample IDs
mapping = load_market_question_mapping()
market_ids = get_sample_market_ids(5)

# Print market questions
for market_id in market_ids:
    print(f"Market: {mapping.get(market_id, 'Unknown')}")
```

## Conclusion

The data infrastructure for this project is designed to handle the complexities of Polymarket data, including blockchain-specific identifiers, multiple tokens per market, and various data formats. The `data_loader.py` script provides a unified interface to access and process this data, abstracting away the underlying complexity and enabling clean, efficient analysis code.
