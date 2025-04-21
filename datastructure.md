# Polymarket Prediction Project Data Structure

## Overview
This document outlines the data structure for the Polymarket prediction project.

## Main Data File
- **Path**: `data/cleaned_election_data.csv`
- **Shape**: (1048575, 54) (rows, columns)
- **Columns**: 
  - Basic market data: `id`, `question`, `slug`, `groupItemTitle`, `startDate`, `endDate`, `description`, `outcomes`, `outcomePrices`, etc.
  - Event metadata: `event_id`, `event_ticker`, `event_slug`, `event_title`, `event_description`, `event_volume`, `event_country`, `event_electionType`, `event_commentCount`
  - Market timeline: `market_start_date`, `market_end_date`
  - Prediction metrics: `correct_outcome`, `yes_token_id`, `closing_price`, `price_2days_prior`, `pre_election_vwap_48h`, etc.
  - Trading metrics: `trading_frequency`, `buy_sell_ratio`, `trading_continuity`, `late_stage_participation`, `volume_acceleration`, etc.
  - Trader metrics: `unique_traders_count`, `trader_to_trade_ratio`, `two_way_traders_ratio`, `trader_concentration`, `new_trader_influx`
  - Community metrics: `comment_per_vol`, `comment_per_trader`
  - Evaluation metrics: `actual_outcome`, `brier_score`, `log_loss`

## Trades Directory Structure
The trades directory contains trading data organized in subdirectories, along with three JSON configuration files:

### JSON Configuration Files

#### 1. collect_summary.json
Contains summary information about token processing:
```json
{
  "timestamp": "2025-03-13T18:35:26.898686",
  "tokens_processed": 1000,
  "tokens_successful": 1000,
  "tokens_with_trades": 978,
  "total_trades": 9596842,
  "results": [
    {
      "token_id": "78118529524639173700167934778533202620053845007966912944800512231568696242280",
      "trade_count": 430,
      "file_path": "polymarket_raw_data/trades/78118529524639173700167934778533202620053845007966912944800512231568696242280.parquet",
      "success": true
    },
    // Additional token entries...
  ]
}
```

#### 2. enhanced_collection_summary.json
Contains detailed information about various data types collected:
```json
{
  "timestamp": "2025-03-14T00:28:25.867876",
  "tokens_processed": 1000,
  "tokens_successful": 1000,
  "collected_data_types": ["orderbooks", "matched_events"],
  "trade_summary": {
    "total_tokens": 0,
    "successful_tokens": 0,
    "skipped_tokens": 0,
    "tokens_with_data": 0,
    "total_trades": 0
  },
  "orderbook_summary": {
    "total_tokens": 1000,
    "successful_tokens": 1000,
    "skipped_tokens": 0,
    "tokens_with_data": 978
  },
  "matched_events_summary": {
    "total_tokens": 1000,
    "successful_tokens": 1000,
    "skipped_tokens": 0,
    "tokens_with_data": 0,
    "total_events": 0
  },
  "token_results": [
    {
      "token_id": "23452090462928163585257733383879365528898800849298930788345778676568194082451",
      "success": true,
      "orderbook_success": true,
      "orderbook_skipped": false,
      "has_orderbook_data": true,
      "events_success": true,
      "events_count": 0,
      "events_skipped": false
    },
    // Additional token entries...
  ]
}
```

#### 3. market_tokens.json
Maps market questions to their associated token IDs:
```json
{
  "Will Donald Trump win the 2024 US Presidential Election?": [
    "21742633143463906290569050155826241533067272736897614950488156847949938836455",
    "48331043336612883890938759509493159234755048973500640148014422747788308965732"
  ],
  "Will Kamala Harris win the 2024 US Presidential Election?": [
    "69236923620077691027083946871148646972011131466059644796654161903044970987404",
    "87584955359245246404952128082451897287778571240979823316620093987046202296181"
  ],
  // Additional market entries...
}
```

### Data Subdirectories
The trades directory contains 4 market subdirectories:
- `matched_events`: Contains parquet files with event data for each market token
- `trades`: Contains trade data for individual tokens
- (Other subdirectories not fully described in the input data)

### Sample File Structure for Parquet Files
- **Format**: Parquet
- **Sample File Schema** (trade files):
  - Columns: 'id', 'timestamp', 'makerAmountFilled', 'takerAmountFilled', 'makerAssetID', 'takerAssetID', 'token_id'

## External Directory
Currently empty directory, possibly reserved for future external data integration.

## Target Variables
The primary evaluation metrics for prediction accuracy are:
- `brier_score`: min=1e-06, max=0.931225, mean=0.0372
- `log_loss`: min=0.0010005, max=3.352407217, mean=0.1216
- `prediction_correct`: min=0.0, max=1.0, mean=0.9447
