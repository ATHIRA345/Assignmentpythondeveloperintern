#Progam genrated for Pythondeveloper intern on 20-03-2026 by Athira Karthik############
"""Bitcoin trader performance analysis against market sentiment.

This script:
1. Loads the Fear & Greed sentiment dataset and Hyperliquid trade history.
2. Cleans and standardizes the real CSV schema used in this assignment.
3. Merges trades with same-day sentiment labels.
4. Computes summary metrics at overall, sentiment, trader, coin, and daily levels.
5. Generates plots for the main trading-performance patterns.
6. Saves all tabular outputs, plots, and a short text report to ``outputs/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
SENTIMENT_PATH = BASE_DIR / "fear_greed_index.csv"
TRADES_PATH = BASE_DIR / "historical_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

SENTIMENT_ORDER = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
]
SENTIMENT_SCORE_MAP = {label: score for score, label in enumerate(SENTIMENT_ORDER)}

"""Create the output folders used by the analysis run.

    Expected output:
    - ``outputs/`` for CSVs and the summary report.
    - ``outputs/plots/`` for PNG chart files.
    """
def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

"""Load the raw sentiment and trade CSV files.

    Returns:
    - A tuple of DataFrames: ``(df_sent, df_trades)``.
    """
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
   
    df_sent = pd.read_csv(SENTIMENT_PATH)
    df_trades = pd.read_csv(TRADES_PATH)
    return df_sent, df_trades

"""Clean and standardize the sentiment dataset.

    Purpose:
    - Normalize column names and data types.
    - Parse the sentiment date field.
    - Add an ordinal sentiment score for numeric analysis.

    Expected output:
    - A cleaned sentiment DataFrame with ``Date`` ready for merge operations.
    """
def prepare_sentiment_data(df_sent: pd.DataFrame) -> pd.DataFrame:
    
    df_sent = df_sent.copy()
    df_sent.columns = [col.strip() for col in df_sent.columns]
    df_sent["date"] = pd.to_datetime(df_sent["date"], errors="coerce")
    df_sent["classification"] = df_sent["classification"].astype(str).str.strip()
    df_sent["value"] = pd.to_numeric(df_sent["value"], errors="coerce")
    df_sent["sentiment_score"] = df_sent["classification"].map(SENTIMENT_SCORE_MAP)
    df_sent = df_sent.dropna(subset=["date", "classification"]).copy()
    df_sent["Date"] = df_sent["date"].dt.date
    return df_sent

"""Clean and standardize the trader dataset.

    Purpose:
    - Parse numeric trade columns safely.
    - Parse ``Timestamp IST`` into a datetime column.
    - Create a date key for merging with sentiment.

    Expected output:
    - A cleaned trade DataFrame with typed numeric columns and a ``Date`` column.
    """
def prepare_trade_data(df_trades: pd.DataFrame) -> pd.DataFrame:
    df_trades = df_trades.copy()
    df_trades.columns = [col.strip() for col in df_trades.columns]

    numeric_columns = [
        "Execution Price",
        "Size Tokens",
        "Size USD",
        "Start Position",
        "Closed PnL",
        "Fee",
        "Timestamp",
    ]
    for col in numeric_columns:
        df_trades[col] = pd.to_numeric(df_trades[col], errors="coerce")

    df_trades["Timestamp IST"] = pd.to_datetime(
        df_trades["Timestamp IST"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    df_trades["Date"] = df_trades["Timestamp IST"].dt.date
    df_trades["Side"] = df_trades["Side"].astype(str).str.strip()
    df_trades["Direction"] = df_trades["Direction"].astype(str).str.strip()
    df_trades["Coin"] = df_trades["Coin"].astype(str).str.strip()
    df_trades["Account"] = df_trades["Account"].astype(str).str.strip()

    df_trades = df_trades.dropna(subset=["Timestamp IST", "Date"]).copy()
    return df_trades

"""Merge trade rows with same-day sentiment values.

    Purpose:
    - Attach sentiment regime and sentiment score to each trade.
    - Add helper flags such as win/loss and absolute PnL.

    Expected output:
    - One merged analysis DataFrame used by all downstream metrics and plots.
    """
def merge_data(df_trades: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        df_trades,
        df_sent[["Date", "classification", "sentiment_score", "value"]],
        on="Date",
        how="left",
    )
    df = df.dropna(subset=["classification"]).copy()
    df["classification"] = pd.Categorical(
        df["classification"],
        categories=SENTIMENT_ORDER,
        ordered=True,
    )
    df["is_profit"] = df["Closed PnL"] > 0
    df["is_loss"] = df["Closed PnL"] < 0
    df["abs_pnl"] = df["Closed PnL"].abs()
    df["notional_to_fee_ratio"] = np.where(
        df["Fee"].fillna(0).ne(0),
        df["Size USD"] / df["Fee"],
        np.nan,
    )
    return df
"""Compute profit factor from a PnL series.

    Purpose:
    - Compare total gains to total losses.

    Expected output:
    - A float value, ``np.inf`` when there are no losses, or ``np.nan`` if undefined.
    """

def compute_profit_factor(series: pd.Series) -> float:
    
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    if losses == 0:
        return np.nan if gains == 0 else np.inf
    return gains / losses

"""Compute a simple risk-adjusted return proxy.

    Purpose:
    - Provide a trade-level Sharpe-like ratio using mean PnL divided by PnL volatility.

    Expected output:
    - A float ratio or ``np.nan`` when the standard deviation is zero or missing.
    """

def compute_sharpe_like(series: pd.Series) -> float:
    
    std = series.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return series.mean() / std

"""Compute drawdown from a cumulative PnL curve.

    Purpose:
    - Measure the decline from the running peak after each trade.

    Expected output:
    - A Series where each value represents drawdown at that point in time.
    """
def compute_drawdown(cumulative_pnl: pd.Series) -> pd.Series:
    
    rolling_peak = cumulative_pnl.cummax()
    return cumulative_pnl - rolling_peak

"""Create the top-level assignment summary metrics.

    Purpose:
    - Summarize activity, profitability, trade size, fees, and drawdown in one table.

    Expected output:
    - A two-column DataFrame with ``Metric`` and ``Value`` fields.
    """

def build_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    
    pnl_series = df["Closed PnL"].dropna()
    cumulative_pnl = pnl_series.cumsum()
    drawdown = compute_drawdown(cumulative_pnl) if not cumulative_pnl.empty else pd.Series(dtype=float)

    metrics = {
        "Total Trades": len(df),
        "Unique Traders": df["Account"].nunique(),
        "Unique Coins": df["Coin"].nunique(),
        "Date Range Start": df["Date"].min(),
        "Date Range End": df["Date"].max(),
        "Total PnL": pnl_series.sum(),
        "Average PnL": pnl_series.mean(),
        "Median PnL": pnl_series.median(),
        "PnL Std Dev": pnl_series.std(),
        "Win Rate": df["is_profit"].mean(),
        "Loss Rate": df["is_loss"].mean(),
        "Profit Factor": compute_profit_factor(pnl_series),
        "Average Win": pnl_series[pnl_series > 0].mean(),
        "Average Loss": pnl_series[pnl_series < 0].mean(),
        "Max Profit Trade": pnl_series.max(),
        "Max Loss Trade": pnl_series.min(),
        "Average Trade Size USD": df["Size USD"].mean(),
        "Median Trade Size USD": df["Size USD"].median(),
        "Average Fee": df["Fee"].mean(),
        "Total Fees": df["Fee"].sum(),
        "Sharpe Like Ratio": compute_sharpe_like(pnl_series),
        "Max Drawdown": drawdown.min() if not drawdown.empty else np.nan,
    }
    return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})


"""Create sentiment-wise performance metrics.

    Purpose:
    - Compare trader behavior and results across the five sentiment regimes.

    Expected output:
    - A DataFrame with one row per sentiment regime and aggregated statistics.
    """
def build_sentiment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    
    rows = []
    for sentiment in SENTIMENT_ORDER:
        subset = df[df["classification"] == sentiment].copy()
        if subset.empty:
            continue
        pnl = subset["Closed PnL"].dropna()
        rows.append(
            {
                "Sentiment": sentiment,
                "Trades": len(subset),
                "Unique_Traders": subset["Account"].nunique(),
                "Total_PnL": pnl.sum(),
                "Avg_PnL": pnl.mean(),
                "Median_PnL": pnl.median(),
                "PnL_Std": pnl.std(),
                "Win_Rate": subset["is_profit"].mean(),
                "Profit_Factor": compute_profit_factor(pnl),
                "Avg_Trade_Size_USD": subset["Size USD"].mean(),
                "Median_Trade_Size_USD": subset["Size USD"].median(),
                "Avg_Fee": subset["Fee"].mean(),
                "Total_Fees": subset["Fee"].sum(),
                "Avg_Sentiment_Index_Value": subset["value"].mean(),
            }
        )
    return pd.DataFrame(rows)
 
"""Create direction distribution metrics by sentiment.

    Purpose:
    - Measure how frequently traders buy or sell in each sentiment regime.

    Expected output:
    - A DataFrame containing count and percentage columns for each direction.
    """

def build_direction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(df["classification"], df["Direction"])
    percentages = pd.crosstab(df["classification"], df["Direction"], normalize="index") * 100
    counts = counts.reindex(SENTIMENT_ORDER).fillna(0)
    percentages = percentages.reindex(SENTIMENT_ORDER).fillna(0)
    output = counts.add_suffix("_Count").join(percentages.add_suffix("_Pct"))
    output.index.name = "Sentiment"
    return output.reset_index()

"""Create account-level trader performance metrics.

    Purpose:
    - Rank traders by profitability and summarize their behavior.

    Expected output:
    - A DataFrame with one row per account sorted by strongest total PnL.
    """
def build_trader_metrics(df: pd.DataFrame) -> pd.DataFrame:
    trader_metrics = (
        df.groupby("Account")
        .agg(
            Trades=("Account", "size"),
            Total_PnL=("Closed PnL", "sum"),
            Avg_PnL=("Closed PnL", "mean"),
            Median_PnL=("Closed PnL", "median"),
            Win_Rate=("is_profit", "mean"),
            Avg_Size_USD=("Size USD", "mean"),
            Total_Fees=("Fee", "sum"),
            Coins_Traded=("Coin", "nunique"),
        )
        .reset_index()
        .sort_values(["Total_PnL", "Win_Rate"], ascending=[False, False])
    )
    return trader_metrics

"""Create coin-level performance metrics.

    Purpose:
    - Compare which coins contributed the most activity and profitability.

    Expected output:
    - A DataFrame with one row per coin sorted by total PnL.
    """

def build_coin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    
    coin_metrics = (
        df.groupby("Coin")
        .agg(
            Trades=("Coin", "size"),
            Total_PnL=("Closed PnL", "sum"),
            Avg_PnL=("Closed PnL", "mean"),
            Win_Rate=("is_profit", "mean"),
            Avg_Size_USD=("Size USD", "mean"),
            Total_Fees=("Fee", "sum"),
            Unique_Traders=("Account", "nunique"),
        )
        .reset_index()
        .sort_values(["Total_PnL", "Trades"], ascending=[False, False])
    )
    return coin_metrics

"""Create daily sentiment-aware performance metrics.

    Purpose:
    - Support time-series analysis for trading activity and market mood.

    Expected output:
    - A DataFrame grouped by ``Date`` and ``classification``.
    """
def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["Date", "classification"], observed=False)
        .agg(
            Trades=("Account", "size"),
            Total_PnL=("Closed PnL", "sum"),
            Avg_PnL=("Closed PnL", "mean"),
            Win_Rate=("is_profit", "mean"),
            Avg_Size_USD=("Size USD", "mean"),
            Avg_Sentiment_Value=("value", "mean"),
        )
        .reset_index()
        .sort_values("Date")
    )
    return daily

"""Create a numeric correlation matrix for key analysis variables.

    Purpose:
    - Show linear relationships between sentiment, price, trade size, fees, and PnL.

    Expected output:
    - A correlation matrix DataFrame ready to export as CSV.
    """
def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[
        [
            "sentiment_score",
            "value",
            "Closed PnL",
            "Execution Price",
            "Size Tokens",
            "Size USD",
            "Fee",
            "Start Position",
        ]
    ].copy()
    return numeric_df.corr(numeric_only=True)


def save_table(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to the main output folder as CSV.

    Expected output:
    - A CSV file written to ``outputs/<filename>``.
    """
    df.to_csv(OUTPUT_DIR / filename, index=False)


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure as a PNG file and close it.

    Expected output:
    - A PNG chart written to ``outputs/plots/<filename>``.
    """
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_avg_pnl_by_sentiment(sentiment_metrics: pd.DataFrame) -> None:
    """Plot average closed PnL for each sentiment regime.

    Plot name:
    - Average PnL by Sentiment

    Expected visual output:
    - A bar chart showing which sentiment regime has the strongest or weakest mean PnL.

    Saved file:
    - ``outputs/plots/avg_pnl_by_sentiment.png``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sentiment_metrics["Sentiment"], sentiment_metrics["Avg_PnL"], color="#4472C4")
    ax.set_title("Average PnL by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Average Closed PnL")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, "avg_pnl_by_sentiment.png")


def plot_win_rate_by_sentiment(sentiment_metrics: pd.DataFrame) -> None:
    """Plot trade win rate for each sentiment regime.

    Plot name:
    - Win Rate by Sentiment

    Expected visual output:
    - A bar chart highlighting the percentage of profitable trades by sentiment regime.

    Saved file:
    - ``outputs/plots/win_rate_by_sentiment.png``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sentiment_metrics["Sentiment"], sentiment_metrics["Win_Rate"] * 100, color="#70AD47")
    ax.set_title("Win Rate by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Win Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, "win_rate_by_sentiment.png")


def plot_pnl_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of closed PnL across sentiment regimes.

    Plot name:
    - PnL Distribution by Sentiment

    Expected visual output:
    - An overlaid histogram showing how gains and losses are distributed in each market mood.

    Saved file:
    - ``outputs/plots/pnl_distribution_by_sentiment.png``
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for sentiment in SENTIMENT_ORDER:
        subset = df.loc[df["classification"] == sentiment, "Closed PnL"].dropna()
        if subset.empty:
            continue
        ax.hist(subset, bins=50, alpha=0.45, label=sentiment)
    ax.set_title("PnL Distribution by Sentiment")
    ax.set_xlabel("Closed PnL")
    ax.set_ylabel("Frequency")
    ax.legend()
    save_plot(fig, "pnl_distribution_by_sentiment.png")


def plot_trade_count_by_sentiment(sentiment_metrics: pd.DataFrame) -> None:
    """Plot the number of trades in each sentiment regime.

    Plot name:
    - Trade Count by Sentiment

    Expected visual output:
    - A bar chart showing whether traders are more active during fear or greed periods.

    Saved file:
    - ``outputs/plots/trade_count_by_sentiment.png``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sentiment_metrics["Sentiment"], sentiment_metrics["Trades"], color="#ED7D31")
    ax.set_title("Trade Count by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Trades")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, "trade_count_by_sentiment.png")


def plot_direction_by_sentiment(direction_metrics: pd.DataFrame) -> None:
    """Plot trade direction mix by sentiment regime.

    Plot name:
    - Direction Distribution by Sentiment

    Expected visual output:
    - A stacked bar chart comparing buy and sell counts across sentiment levels.

    Saved file:
    - ``outputs/plots/direction_by_sentiment.png``
    """
    directions = [col for col in direction_metrics.columns if col.endswith("_Count")]
    if not directions:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(direction_metrics))
    for column in directions:
        values = direction_metrics[column].to_numpy()
        ax.bar(direction_metrics["Sentiment"], values, bottom=bottom, label=column.replace("_Count", ""))
        bottom += values
    ax.set_title("Direction Distribution by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Trade Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    save_plot(fig, "direction_by_sentiment.png")


def plot_size_vs_pnl(df: pd.DataFrame) -> None:
    """Plot trade size against closed PnL, colored by sentiment.

    Plot name:
    - Trade Size vs Closed PnL

    Expected visual output:
    - A scatter plot showing whether larger positions tend to produce larger gains or losses.

    Saved file:
    - ``outputs/plots/size_vs_pnl.png``
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "Extreme Fear": "#264653",
        "Fear": "#2A9D8F",
        "Neutral": "#E9C46A",
        "Greed": "#F4A261",
        "Extreme Greed": "#E76F51",
    }
    for sentiment in SENTIMENT_ORDER:
        subset = df[df["classification"] == sentiment]
        if subset.empty:
            continue
        ax.scatter(
            subset["Size USD"],
            subset["Closed PnL"],
            alpha=0.25,
            s=12,
            label=sentiment,
            color=colors[sentiment],
        )
    ax.set_title("Trade Size vs Closed PnL")
    ax.set_xlabel("Size USD")
    ax.set_ylabel("Closed PnL")
    ax.legend(markerscale=1.5)
    save_plot(fig, "size_vs_pnl.png")


def plot_cumulative_pnl(df: pd.DataFrame) -> None:
    """Plot cumulative PnL over time.

    Plot name:
    - Cumulative PnL Over Time

    Expected visual output:
    - A time-series line chart showing the overall equity curve from sequential trades.

    Saved file:
    - ``outputs/plots/cumulative_pnl_over_time.png``
    """
    df_sorted = df.sort_values("Timestamp IST").copy()
    df_sorted["cum_pnl"] = df_sorted["Closed PnL"].fillna(0).cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_sorted["Timestamp IST"], df_sorted["cum_pnl"], color="#4472C4", linewidth=1.5)
    ax.set_title("Cumulative PnL Over Time")
    ax.set_xlabel("Timestamp IST")
    ax.set_ylabel("Cumulative PnL")
    save_plot(fig, "cumulative_pnl_over_time.png")


def plot_drawdown(df: pd.DataFrame) -> None:
    """Plot the drawdown curve derived from cumulative PnL.

    Plot name:
    - Drawdown Over Time

    Expected visual output:
    - A time-series line chart showing the depth of declines from prior cumulative PnL peaks.

    Saved file:
    - ``outputs/plots/drawdown_over_time.png``
    """
    df_sorted = df.sort_values("Timestamp IST").copy()
    df_sorted["cum_pnl"] = df_sorted["Closed PnL"].fillna(0).cumsum()
    df_sorted["drawdown"] = compute_drawdown(df_sorted["cum_pnl"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_sorted["Timestamp IST"], df_sorted["drawdown"], color="#C00000", linewidth=1.5)
    ax.set_title("Drawdown Over Time")
    ax.set_xlabel("Timestamp IST")
    ax.set_ylabel("Drawdown")
    save_plot(fig, "drawdown_over_time.png")


def plot_daily_activity_vs_sentiment(daily_metrics: pd.DataFrame) -> None:
    """Plot daily trading activity together with the Fear & Greed index.

    Plot name:
    - Daily Trade Activity vs Sentiment Index

    Expected visual output:
    - A dual-axis line chart comparing daily trade count against average daily sentiment value.

    Saved file:
    - ``outputs/plots/daily_activity_vs_sentiment.png``
    """
    daily_totals = (
        daily_metrics.groupby("Date")
        .agg(Trades=("Trades", "sum"), Avg_Sentiment_Value=("Avg_Sentiment_Value", "mean"))
        .reset_index()
        .sort_values("Date")
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(daily_totals["Date"], daily_totals["Trades"], color="#4472C4", label="Trades")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Trades", color="#4472C4")
    ax1.tick_params(axis="y", labelcolor="#4472C4")

    ax2 = ax1.twinx()
    ax2.plot(daily_totals["Date"], daily_totals["Avg_Sentiment_Value"], color="#ED7D31", label="Sentiment Index")
    ax2.set_ylabel("Fear & Greed Index", color="#ED7D31")
    ax2.tick_params(axis="y", labelcolor="#ED7D31")

    fig.suptitle("Daily Trade Activity vs Sentiment Index")
    save_plot(fig, "daily_activity_vs_sentiment.png")


def plot_coin_sentiment_heatmap(df: pd.DataFrame) -> None:
    """Plot average coin-level PnL by sentiment for the most active coins.

    Plot name:
    - Average PnL Heatmap for Top Coins by Sentiment

    Expected visual output:
    - A heatmap where warmer or cooler cells make coin-sentiment performance differences easy to spot.

    Saved file:
    - ``outputs/plots/coin_sentiment_heatmap.png``
    """
    top_coins = df["Coin"].value_counts().head(10).index
    heatmap_df = (
        df[df["Coin"].isin(top_coins)]
        .pivot_table(
            index="Coin",
            columns="classification",
            values="Closed PnL",
            aggfunc="mean",
            observed=False,
        )
        .reindex(columns=SENTIMENT_ORDER)
    )
    if heatmap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(heatmap_df.fillna(0).to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_title("Average PnL Heatmap for Top Coins by Sentiment")
    fig.colorbar(image, ax=ax, label="Average Closed PnL")
    save_plot(fig, "coin_sentiment_heatmap.png")


def build_text_report(
    overall_metrics: pd.DataFrame,
    sentiment_metrics: pd.DataFrame,
    trader_metrics: pd.DataFrame,
    coin_metrics: pd.DataFrame,
) -> str:
    """Build a concise human-readable summary for the assignment.

    Purpose:
    - Convert the most important metrics into a short report suitable for submission review.

    Expected output:
    - A string that will be saved as ``outputs/analysis_report.txt``.
    """
    metric_lookup = dict(zip(overall_metrics["Metric"], overall_metrics["Value"]))
    best_sentiment = sentiment_metrics.sort_values("Avg_PnL", ascending=False).head(1)
    best_win_rate = sentiment_metrics.sort_values("Win_Rate", ascending=False).head(1)
    top_trader = trader_metrics.head(1)
    top_coin = coin_metrics.head(1)

    lines = [
        "Bitcoin Trader Performance vs Market Sentiment",
        "=" * 48,
        "",
        "Overall Summary",
        f"- Total trades analyzed: {int(metric_lookup['Total Trades'])}",
        f"- Unique traders: {int(metric_lookup['Unique Traders'])}",
        f"- Total PnL: {float(metric_lookup['Total PnL']):.2f}",
        f"- Average PnL per trade: {float(metric_lookup['Average PnL']):.4f}",
        f"- Win rate: {float(metric_lookup['Win Rate']) * 100:.2f}%",
        f"- Profit factor: {float(metric_lookup['Profit Factor']):.4f}" if pd.notna(metric_lookup["Profit Factor"]) and np.isfinite(metric_lookup["Profit Factor"]) else "- Profit factor: undefined or infinite",
        f"- Max drawdown: {float(metric_lookup['Max Drawdown']):.2f}" if pd.notna(metric_lookup["Max Drawdown"]) else "- Max drawdown: not available",
        "",
        "Key Findings",
    ]

    if not best_sentiment.empty:
        row = best_sentiment.iloc[0]
        lines.append(
            f"- Highest average PnL sentiment: {row['Sentiment']} ({row['Avg_PnL']:.4f})"
        )
    if not best_win_rate.empty:
        row = best_win_rate.iloc[0]
        lines.append(
            f"- Highest win rate sentiment: {row['Sentiment']} ({row['Win_Rate'] * 100:.2f}%)"
        )
    if not top_trader.empty:
        row = top_trader.iloc[0]
        lines.append(
            f"- Top trader by total PnL: {row['Account']} ({row['Total_PnL']:.2f})"
        )
    if not top_coin.empty:
        row = top_coin.iloc[0]
        lines.append(
            f"- Top coin by total PnL: {row['Coin']} ({row['Total_PnL']:.2f})"
        )

    lines.extend(
        [
            "",
            "Generated Files",
            "- merged_trader_sentiment.csv",
            "- overall_metrics.csv",
            "- sentiment_metrics.csv",
            "- direction_metrics.csv",
            "- trader_metrics.csv",
            "- coin_metrics.csv",
            "- daily_metrics.csv",
            "- correlation_matrix.csv",
            "- plots/*.png",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the full end-to-end analysis workflow.

    Flow:
    - Create folders.
    - Load and clean both datasets.
    - Merge data on date.
    - Build metrics tables.
    - Save CSV outputs.
    - Generate and save all plots.
    - Write the final text report.

    Expected output:
    - A populated ``outputs/`` directory containing all analysis artifacts.
    """
    ensure_output_dirs()

    df_sent_raw, df_trades_raw = load_data()
    df_sent = prepare_sentiment_data(df_sent_raw)
    df_trades = prepare_trade_data(df_trades_raw)
    df = merge_data(df_trades, df_sent)

    overall_metrics = build_overall_metrics(df)
    sentiment_metrics = build_sentiment_metrics(df)
    direction_metrics = build_direction_metrics(df)
    trader_metrics = build_trader_metrics(df)
    coin_metrics = build_coin_metrics(df)
    daily_metrics = build_daily_metrics(df)
    correlation_matrix = build_correlation_matrix(df)

    # Save the cleaned merged master table used for all analysis.
    save_table(df, "merged_trader_sentiment.csv")
    save_table(overall_metrics, "overall_metrics.csv")
    save_table(sentiment_metrics, "sentiment_metrics.csv")
    save_table(direction_metrics, "direction_metrics.csv")
    save_table(trader_metrics, "trader_metrics.csv")
    save_table(coin_metrics, "coin_metrics.csv")
    save_table(daily_metrics, "daily_metrics.csv")
    correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Plot 1: Mean profitability under each sentiment label.
    plot_avg_pnl_by_sentiment(sentiment_metrics)
    # Plot 2: Share of profitable trades by sentiment.
    plot_win_rate_by_sentiment(sentiment_metrics)
    # Plot 3: Distribution of gains and losses across sentiment regimes.
    plot_pnl_distribution(df)
    # Plot 4: Number of trades executed in each sentiment regime.
    plot_trade_count_by_sentiment(sentiment_metrics)
    # Plot 5: Buy versus sell behavior under each sentiment regime.
    plot_direction_by_sentiment(direction_metrics)
    # Plot 6: Relationship between trade size and realized PnL.
    plot_size_vs_pnl(df)
    # Plot 7: Running cumulative PnL over time.
    plot_cumulative_pnl(df)
    # Plot 8: Decline from prior cumulative PnL peaks.
    plot_drawdown(df)
    # Plot 9: Daily trade count compared with the fear-greed index level.
    plot_daily_activity_vs_sentiment(daily_metrics)
    # Plot 10: Coin-wise profitability differences across sentiment regimes.
    plot_coin_sentiment_heatmap(df)

    report_text = build_text_report(overall_metrics, sentiment_metrics, trader_metrics, coin_metrics)
    (OUTPUT_DIR / "analysis_report.txt").write_text(report_text, encoding="utf-8")

    print("Analysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
