# Bitcoin Trader Performance vs Market Sentiment

## Overview

This project analyzes the relationship between Bitcoin market sentiment and trader performance using two datasets:

1. `fear_greed_index.csv`
   - Contains the Bitcoin Fear & Greed Index values and sentiment labels.
2. `historical_data.csv`
   - Contains historical Hyperliquid trader execution data, including trade direction, trade size, fees, and realized PnL.

The main objective is to explore whether trader behavior and profitability change under different market sentiment regimes such as `Extreme Fear`, `Fear`, `Neutral`, `Greed`, and `Extreme Greed`.


## Project Files

- [bitcoin_sentiment_analysis.py](basepath+\bitcoin_sentiment_analysis.py)
  - Main analysis script.
- [fear_greed_index.csv](basepath +\fear_greed_index.csv)
  - Sentiment dataset.
- [historical_data.csv](basepath+\historical_data.csv)
  - Trader dataset.


## Dataset Details

### 1. Sentiment Dataset

File: `fear_greed_index.csv`

Main columns:
- `date`
- `classification`
- `value`
- `timestamp`

This dataset provides the daily market sentiment label and index value.


### 2. Trader Dataset

File: `historical_data.csv`

Main columns:
- `Account`
- `Coin`
- `Execution Price`
- `Size Tokens`
- `Size USD`
- `Side`
- `Direction`
- `Timestamp IST`
- `Start Position`
- `Closed PnL`
- `Fee`
- `Order ID`
- `Trade ID`
- `Transaction Hash`

This dataset provides trade-level execution and realized profitability data.


## Analysis Workflow

The script follows this flow:

1. Load both CSV files.
2. Clean and standardize date and numeric fields.
3. Convert trade timestamps into calendar dates.
4. Merge each trade with the same-day sentiment record.
5. Engineer analysis fields such as:
   - `is_profit`
   - `is_loss`
   - `abs_pnl`
   - `sentiment_score`
6. Generate summary metrics.
7. Create plots to visualize behavior and performance patterns.
8. Save all outputs automatically into the `outputs/` folder.


## Metrics Generated

The script computes the following types of metrics:

### Overall Metrics
- Total trades
- Unique traders
- Unique coins
- Total PnL
- Average PnL
- Median PnL
- PnL standard deviation
- Win rate
- Loss rate
- Profit factor
- Average win
- Average loss
- Maximum profit trade
- Maximum loss trade
- Average trade size
- Average fee
- Total fees
- Sharpe-like ratio
- Maximum drawdown

### Sentiment-wise Metrics
- Trades per sentiment
- Unique traders per sentiment
- Total PnL by sentiment
- Average PnL by sentiment
- Median PnL by sentiment
- PnL volatility by sentiment
- Win rate by sentiment
- Profit factor by sentiment
- Average trade size by sentiment
- Average fee by sentiment

### Trader-wise Metrics
- Trades by account
- Total PnL by account
- Average PnL by account
- Win rate by account
- Average trade size by account
- Total fees by account

### Coin-wise Metrics
- Trades by coin
- Total PnL by coin
- Average PnL by coin
- Win rate by coin
- Average trade size by coin
- Total fees by coin

### Daily Metrics
- Trades per day
- Total daily PnL
- Average daily PnL
- Daily win rate
- Average daily trade size
- Average daily sentiment index value


## Plots Generated

The script saves the following charts inside `outputs/plots/`:

1. `avg_pnl_by_sentiment.png`
   - Average realized PnL for each sentiment regime.
2. `win_rate_by_sentiment.png`
   - Percentage of profitable trades by sentiment.
3. `pnl_distribution_by_sentiment.png`
   - Distribution of trade-level PnL for each sentiment.
4. `trade_count_by_sentiment.png`
   - Number of trades in each sentiment regime.
5. `direction_by_sentiment.png`
   - Buy and sell trade mix across sentiment regimes.
6. `size_vs_pnl.png`
   - Scatter plot of trade size versus realized PnL.
7. `cumulative_pnl_over_time.png`
   - Running cumulative PnL over time.
8. `drawdown_over_time.png`
   - Drawdown curve derived from cumulative PnL.
9. `daily_activity_vs_sentiment.png`
   - Daily trade activity compared to the Fear & Greed index.
10. `coin_sentiment_heatmap.png`
   - Average PnL heatmap for top coins across sentiment regimes.


## Output Files

After running the script, the following files are created in `outputs/`:

- `merged_trader_sentiment.csv`
- `overall_metrics.csv`
- `sentiment_metrics.csv`
- `direction_metrics.csv`
- `trader_metrics.csv`
- `coin_metrics.csv`
- `daily_metrics.csv`
- `correlation_matrix.csv`
- `analysis_report.txt`

Plot images are saved in:

- `outputs/plots/`


## How To Run

Run the script from the project folder:

```powershell
python bitcoin_sentiment_analysis.py
```


## Required Python Libraries

Install the required libraries before running:

```powershell
pip install pandas numpy matplotlib
```


## Expected Final Output

At the end of execution, the project should provide:

- A merged trade-sentiment dataset
- Summary CSV files for analysis
- Visualization charts for key trading patterns
- A short text report highlighting the major findings

This output can be used directly for:
- assignment submission
- business insight reporting
- trading strategy exploration
- sentiment-based performance comparison



## Submission Summary

This project delivers a complete Python-based analysis pipeline for evaluating the impact of Bitcoin market sentiment on trader performance. It combines data cleaning, feature engineering, performance metrics, visual exploration, and exportable outputs in a single reusable script.
