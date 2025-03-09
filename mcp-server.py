from typing import Any
import os
import httpx
from mcp.server.fastmcp import FastMCP
import pandas as pd
import datetime
from ta.utils import dropna
from ta.trend import EMAIndicator

from dotenv import load_dotenv
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("weather")

# DB Class
from pymongo import MongoClient

class MongoDB:
    client = None
    db = None

    @classmethod
    def init_client(cls):
        if not cls.client:
            cls.client = MongoClient(os.getenv("MONGODB_URI"))
            cls.db = cls.client.lumifai

    @classmethod
    def get_db(cls):
        if not cls.client:
            cls.init_client()
        return cls.db

    @classmethod
    def close_client(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None

def fetch_agents_like_name(partial_name):
    db = MongoDB.get_db()
    collection = db.ai_agents # type: ignore

    pipeline = [
        {
            '$search': {
                'index': 'default_text_index',
                'phrase': {
                    'query': partial_name,
                    'path': ['agent_name', 'base_token_symbol']
                }
            }
        }
    ]

    agents = list(collection.aggregate(pipeline))
    return agents

# Helper functions
def fetch_binance_ohlcv_data(agent: dict, time_ago: str, interval: int, interval_frequency: str) -> pd.DataFrame:
    """Fetch OHLCV data from MongoDB for Binance pairs"""
    try:
        db = MongoDB.get_db()
        collection = db.ohlcv_data  # type: ignore

        # Convert time_ago to datetime
        since_time = pd.to_datetime(time_ago)

        # Construct the symbol from base and quote tokens
        symbol = f"{agent['base_token_symbol']}{agent['quote_token_symbol']}"

        # Map interval and frequency to Binance format (e.g., '1m', '1h', '1d')
        interval_map = {
            'minutes': 'm',
            'hours': 'h',
            'days': 'd',
            'weeks': 'w',
            'months': 'M'
        }
        binance_interval = f"{interval}{interval_map.get(interval_frequency, 'm')}"

        # Query MongoDB
        pipeline = [
            {
                "$match": {
                    "metadata.symbol": symbol,
                    "metadata.interval": binance_interval,
                    "t": {"$gte": since_time}
                }
            },
            {
                "$project": {
                    "time": {"$toLong": "$t"},
                    "open": "$o",
                    "high": "$h",
                    "low": "$l",
                    "close": "$c",
                    "volume": "$v"
                }
            },
            {
                "$sort": {"time": 1}
            }
        ]

        results = list(collection.aggregate(pipeline))

        if not results:
            raise Exception(f"No data found for symbol {symbol} with interval {binance_interval}")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert timestamp to seconds
        df['time'] = df['time'] // 1000  # Convert milliseconds to seconds

        return df
    except Exception as e:
        logger.error(f"Error fetching Binance OHLCV data: {str(e)}")
        raise
    finally:
        logger.info(f"Completed Binance OHLCV data fetch attempt for symbol {agent['base_token_symbol']}")

@mcp.tool()
def get_emas(agent_name: str, time_ago: str, interval: int, interval_frequency: str) -> pd.DataFrame:
    """
    Calculate the Exponential Moving Average (EMA) of the given data. Returns the dataframe with two new columns added trend_ema_fast calculated with 12 periods and trend_ema_slow calculated with 26 periods.

    Args:
      agent_name (str): Name of the agent token for which the OHCLV data is to be fetched
      time_ago (datetime): Time since you want the historic data from as an ISO Date string (eg. 2019-08-31T15:47:06Z)
      interval (int): Interval of the OHCLV data to query
      interval_frequency (str): Frequency of the interval (eg. minutes, hours, days, weeks, months, years)
    """
    import time

    start_time = time.time()
    logger.info(f"Starting EMA calculation for {agent_name}")
    # First, get the details from the sql table
    agents_fetched = fetch_agents_like_name(agent_name)

    # If there are no agents with that name, return an error
    if len(agents_fetched) == 0:
        raise Exception(f"No agents found with the name {agent_name}")

    # Use the first agent retrieved
    agent = agents_fetched[0]
    df = fetch_binance_ohlcv_data(agent, time_ago, interval, interval_frequency)

    if 'Exception' in df.columns:
        execution_time = time.time() - start_time
        error_msg = df['Exception'].iloc[0]
        logger.error(f"Error in EMA calculation after {execution_time:.2f} seconds: {error_msg}")
        return error_msg

    df["volume"] = df["volume"].map('{:.10f}'.format)
    try:
        df["trend_ema_fast"] = EMAIndicator(
            close=df["close"], window=12, fillna=True
        ).ema_indicator()
        df["trend_ema_slow"] = EMAIndicator(
            close=df["close"], window=26, fillna=True
        ).ema_indicator()
        df = dropna(df)
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in EMA calculation after {execution_time:.2f} seconds: {str(e)}")
        return str(e)

    execution_time = time.time() - start_time
    logger.info(f"EMA calculation completed in {execution_time:.2f} seconds")
    return df

@mcp.tool()
def get_date_time() -> datetime.datetime:
    """Returns the current date and time"""
    return datetime.datetime.now()

if __name__ == "__main__":
    # Initialize and run the server
    try:
        mcp.run(transport='sse')
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
