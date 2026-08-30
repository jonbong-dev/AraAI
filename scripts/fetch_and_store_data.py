#!/usr/bin/env python3
"""
Fetch market data and store in SQLite database.
Uses yf.download batching with multi-threading to prevent Yahoo Finance IP rate limits in CI runners.
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

STOCK_TIMEFRAMES = [
    ("max", "1d"),
]

FOREX_TIMEFRAMES = [
    ("max", "1d"),
]


def init_database(db_file):
    """Initialize database schema if it doesn't exist."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            asset_type TEXT,
            timeframe TEXT,
            interval TEXT,
            fetch_date TEXT,
            UNIQUE(symbol, date, interval)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            model_type TEXT,
            training_date TEXT,
            accuracy REAL,
            loss REAL,
            epochs INTEGER,
            model_path TEXT,
            timeframe TEXT,
            training_mode TEXT,
            hour INTEGER
        )
    """)

    conn.commit()
    conn.close()


def fetch_and_store_batch(symbols, db_file, asset_type, period="max", interval="1d"):
    """Fetch all symbols in parallel via yf.download to bypass rate-limiting."""
    formatted_symbols = [
        f"{s}=X" if (asset_type == "forex" and not s.endswith("=X")) else s
        for s in symbols
    ]

    print(f"Batch fetching {len(formatted_symbols)} {asset_type} symbols via yfinance...")
    try:
        data = yf.download(
            tickers=formatted_symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as e:
        print(f"Error executing batch download: {e}")
        return 0, 0

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    total_rows = 0
    successful = 0
    fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for raw_symbol, ticker_symbol in zip(symbols, formatted_symbols):
        try:
            if len(formatted_symbols) == 1:
                df = data.copy()
            else:
                if ticker_symbol not in data.columns.levels[0]:
                    print(f"  [SKIP] {raw_symbol}: Not returned by Yahoo Finance")
                    continue
                df = data[ticker_symbol].dropna(how="all").copy()

            if df.empty:
                print(f"  [SKIP] {raw_symbol}: Dataset empty")
                continue

            df = df.reset_index()
            date_col = "Date" if "Date" in df.columns else "Datetime"

            rows_added = 0
            for _, row in df.iterrows():
                if str(row["Close"]) == "nan":
                    continue

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO market_data
                    (symbol, date, open, high, low, close, volume, asset_type, timeframe, interval, fetch_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw_symbol,
                        (
                            row[date_col].strftime("%Y-%m-%d %H:%M:%S")
                            if hasattr(row[date_col], "strftime")
                            else str(row[date_col])
                        ),
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row.get("Volume", 0)),
                        asset_type,
                        period,
                        interval,
                        fetch_time,
                    ),
                )
                rows_added += 1

            if rows_added > 0:
                total_rows += rows_added
                successful += 1
                print(f"  [OK] {raw_symbol}: {rows_added} rows")

        except Exception as e:
            print(f"  [ERROR] {raw_symbol}: {e}")

    conn.commit()
    conn.close()
    return total_rows, successful


def main():
    parser = argparse.ArgumentParser(description="Fetch and store market data")
    parser.add_argument("--db-file", required=True, help="Database file path")
    parser.add_argument("--asset-type", choices=["stock", "forex"], required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Fetch only the first N symbols of the list",
    )
    parser.add_argument("--period", default=None, help="Data period")
    parser.add_argument("--interval", default=None, help="Data interval")

    args = parser.parse_args()

    init_database(args.db_file)

    if args.asset_type == "stock":
        symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT",
            "JNJ", "PG", "MA", "UNH", "HD", "DIS", "BAC", "VZ", "ADBE", "CMCSA",
            "NFLX", "PFE", "INTC", "KO", "PEP", "CSCO", "ABT", "CRM", "T", "ABBV",
            "CVX", "NKE", "MRK", "MCD", "MDT", "TXN", "HON", "BA", "UNP", "AMGN",
            "IBM", "QCOM", "ORCL", "SBUX", "GS", "MMM", "CAT", "GE", "F", "GM",
            "C", "TGT", "LMT", "DE", "LOW", "UPS", "USB", "AXP", "MS", "WFC",
            "COP", "SLB", "EOG", "OXY", "VLO", "MPC", "PSX", "KMI", "WMB", "NEE",
            "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "WEC", "AMT",
            "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "SPG", "AVB", "VRTX",
            "REGN", "ISRG", "SYK", "ZTS", "BSX", "EW", "GILD", "BIIB", "ILMN", "AMD",
            "WDC", "MU", "STX", "NOK", "LRCX"
        ]
        timeframes = STOCK_TIMEFRAMES
    else:
        symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP",
            "EURJPY", "GBPJPY", "CHFJPY", "EURCHF", "AUDJPY", "NZDJPY", "CADJPY",
            "EURAUD", "EURCAD", "GBPAUD", "GBPCAD", "AUDCAD", "AUDNZD", "EURNZD", "GBPNZD"
        ]
        timeframes = FOREX_TIMEFRAMES

    if args.limit:
        symbols = symbols[: args.limit]

    period = args.period if args.period else timeframes[0][0]
    interval = args.interval if args.interval else timeframes[0][1]

    total_rows, successful = fetch_and_store_batch(
        symbols, args.db_file, args.asset_type, period, interval
    )

    print(f"\nSummary: {successful}/{len(symbols)} symbols, {total_rows} total rows stored")

    if total_rows == 0:
        print("[WARNING] No data stored")
        sys.exit(0)


if __name__ == "__main__":
    main()
