import requests
import pandas as pd
import numpy as np
import time
import asyncio
import aiohttp
import logging
from typing import List, Tuple, Optional, Generator, Dict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import date, timedelta
from calendar import monthrange
import datetime
import pytz
import streamlit as st
import csv
import warnings
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Streamlit page configuration
st.set_page_config(page_title="Option Chain Analysis Dashboard", layout="wide", menu_items=None)

# Custom CSS
def load_custom_css():
    return """
    <style>
        .block-container {
            padding: 1rem 5rem 0rem 5rem;
        }
        .metric-card {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 1rem;
            color: white;
        }
        .data-table {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """

st.markdown(load_custom_css(), unsafe_allow_html=True)

# Configuration
@dataclass
class Config:
    EXCHANGE: str = "NSE"
    PRICE_OFFSET_PERCENT: float = 0.015
    STRIKE_RANGE: int = 5
    REFRESH_INTERVAL: int = 5
    CACHE_DURATION: int = 30  # seconds

# Enums
class HighlightThreshold(Enum):
    PREMIUM = 1.0
    PREMIUM_SP = 5.0

class HighlightColor(Enum):
    PREMIUM = 'background-color: paleturquoise'
    PREMIUM_SP = 'background-color: wheat'
    DEFAULT = ''

# Data classes
@dataclass
class PriceRange:
    low: float
    high: float
    
    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("Low price cannot be higher than high price")

@dataclass
class OptionData:
    strike_price: float
    expiry_date: str
    last_price: float
    instrument_type: str

@dataclass
class TableConfig:
    table_number: int
    selected_option: str
    exp_option: str

# Utility functions
def get_last_thursdays(year: int) -> List[date]:
    """Calculate the last Thursday of each month for the given year."""
    expiry_dates = []
    
    for month in range(1, 13):
        # Get the last day of the month
        _, last_day = monthrange(year, month)
        last_date = date(year, month, last_day)
        
        # Find the last Thursday
        days_back = (last_date.weekday() - 3) % 7
        last_thursday = last_date - timedelta(days=days_back)
        
        expiry_dates.append(last_thursday)
    
    return expiry_dates

def get_future_expiry_dates() -> Tuple[List[str], str]:
    """Get future expiry dates and default expiry option."""
    today = datetime.date.today()
    current_year = today.year
    
    # Get all expiry dates for current year
    expiry_dates = get_last_thursdays(current_year)
    
    # Filter future dates and format
    future_dates = [
        date_obj.strftime('%d-%m-%Y')
        for date_obj in expiry_dates
        if (date_obj - today).days >= 0
    ]
    
    # If no future dates in current year, get next year
    if not future_dates:
        next_year_dates = get_last_thursdays(current_year + 1)
        future_dates = [
            date_obj.strftime('%d-%m-%Y')
            for date_obj in next_year_dates
            if (date_obj - today).days >= 0
        ]
    
    default_expiry = future_dates[0] if future_dates else "31-12-2025"
    
    return future_dates, default_expiry

# Initialize global variables
DATE_LIST, EXP_OPTION = get_future_expiry_dates()

# Market Data Provider
class MarketDataProvider:
    def __init__(self):
        self.session = None
        self.price_cache = {}
        self.cache_duration = Config.CACHE_DURATION
    
    async def get_current_price(self, ticker: str, exchange: str) -> Optional[float]:
        """Get current market price with caching and error handling."""
        cache_key = f"{ticker}:{exchange}"
        
        # Check cache first
        if cache_key in self.price_cache:
            timestamp, price = self.price_cache[cache_key]
            if (datetime.datetime.now() - timestamp).seconds < self.cache_duration:
                return price
        
        try:
            url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Try multiple CSS selectors for robustness
                    price_selectors = [
                        "YMlKec fxKbKc",
                        "fxKbKc",
                        "[data-last-price]"
                    ]
                    
                    for selector in price_selectors:
                        element = soup.find(class_=selector)
                        if element:
                            price_text = element.text.strip()
                            if price_text.startswith('₹'):
                                price = float(price_text[1:].replace(",", ""))
                                self.price_cache[cache_key] = (datetime.datetime.now(), price)
                                return price
                    
                    raise ValueError(f"Price not found for {ticker}")
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

    def get_current_price_sync(self, ticker: str, exchange: str) -> Optional[float]:
        """Synchronous version for compatibility."""
        try:
            url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            element = soup.find(class_="YMlKec fxKbKc")
            
            if element:
                price_text = element.text.strip()
                if price_text.startswith('₹'):
                    price = float(price_text[1:].replace(",", ""))
                    return price
            
            return None
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

# 52-week range functions
def get_52_week_range(ticker: str, exchange: str) -> PriceRange:
    """Get 52-week high/low prices with robust error handling."""
    try:
        url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple selectors for 52-week range
        range_selectors = [
            "P6K39c",
            "[data-52-week-range]",
            ".range-data"
        ]
        
        for selector in range_selectors:
            elements = soup.find_all(class_=selector)
            for element in elements:
                text = element.text.strip()
                if "-" in text and "₹" in text:
                    # Parse range like "₹1,000 - ₹2,000"
                    parts = text.split("-")
                    if len(parts) == 2:
                        low_str = parts[0].strip().replace("₹", "").replace(",", "")
                        high_str = parts[1].strip().replace("₹", "").replace(",", "")
                        
                        try:
                            low = float(low_str)
                            high = float(high_str)
                            return PriceRange(low, high)
                        except ValueError:
                            continue
        
        raise ValueError(f"52-week range not found for {ticker}")
        
    except Exception as e:
        logger.error(f"Error fetching 52-week range for {ticker}: {str(e)}")
        return PriceRange(0.0, 0.0)

# Highlighting functions
def get_highlight_style(value: float, column_name: str) -> str:
    """Get highlight style based on value and column type."""
    if not isinstance(value, (int, float)):
        return HighlightColor.DEFAULT.value
    
    # Define highlighting rules
    highlight_rules = {
        "CE Premium%": (HighlightThreshold.PREMIUM.value, HighlightColor.PREMIUM.value),
        "PE Premium%": (HighlightThreshold.PREMIUM.value, HighlightColor.PREMIUM.value),
        "CE (Premium+SP)%": (HighlightThreshold.PREMIUM_SP.value, HighlightColor.PREMIUM_SP.value),
        "PE (Premium+SP)%": (HighlightThreshold.PREMIUM_SP.value, HighlightColor.PREMIUM_SP.value),
    }
    
    threshold, color = highlight_rules.get(column_name, (0, HighlightColor.DEFAULT.value))
    return color if value > threshold else HighlightColor.DEFAULT.value

# Option Chain Processor
class OptionChainProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get_nse_cookies(self) -> dict:
        """Get NSE cookies for API access."""
        try:
            response = self.session.get("https://www.nseindia.com/")
            return response.cookies
        except Exception as e:
            logger.error(f"Error getting NSE cookies: {e}")
            return {}
    
    def _fetch_option_chain_data(self, ticker: str) -> Optional[dict]:
        """Fetch option chain data from NSE API."""
        try:
            cookies = self._get_nse_cookies()
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
            
            response = self.session.get(url, cookies=cookies, timeout=15)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching option chain for {ticker}: {e}")
            return None
    
    def _process_option_data(self, raw_data: dict) -> pd.DataFrame:
        """Process raw option chain data into structured format."""
        option_data = []
        
        for record in raw_data.get("records", {}).get("data", []):
            for option_type, data in record.items():
                if option_type in ["CE", "PE"]:
                    data["instrumentType"] = option_type
                    option_data.append(data)
        
        return pd.DataFrame(option_data)
    
    def _calculate_atm_strikes(self, df: pd.DataFrame, current_price: float) -> Tuple[float, float]:
        """Calculate ATM strikes for CE and PE options."""
        strikes = sorted(df['strikePrice'].unique())
        if len(strikes) < 2:
            return current_price, current_price
        
        strike_size = strikes[1] - strikes[0]
        
        # Calculate 2% above and below current price
        ce_target = current_price * (1 + Config.PRICE_OFFSET_PERCENT)
        pe_target = current_price * (1 - Config.PRICE_OFFSET_PERCENT)
        
        # Find nearest strikes
        atm_ce = round(ce_target / strike_size) * strike_size
        atm_pe = round(pe_target / strike_size) * strike_size
        
        return atm_ce, atm_pe
    
    def _get_strikes_range(self, df: pd.DataFrame, atm_strike: float, 
                          strike_size: float, is_ce: bool = True) -> List[float]:
        """Get range of strikes around ATM."""
        strikes = []
        current_strike = atm_strike
        
        for _ in range(Config.STRIKE_RANGE):
            if current_strike in df['strikePrice'].values:
                strikes.append(current_strike)
            
            if is_ce:
                current_strike += strike_size
            else:
                current_strike -= strike_size
        
        return strikes
    
    def _format_expiry_date(self, date_str: str) -> str:
        """Convert expiry date format."""
        try:
            parsed_date = datetime.datetime.strptime(date_str, '%d-%b-%Y')
            return parsed_date.strftime('%d-%m-%Y')
        except ValueError:
            return date_str
    
    def get_option_chain(self, ticker: str, expiry_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method to get processed option chain data."""
        # Fetch raw data
        raw_data = self._fetch_option_chain_data(ticker)
        if not raw_data:
            return pd.DataFrame(), pd.DataFrame()
        
        # Process data
        df = self._process_option_data(raw_data)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Get current price
        market_data = MarketDataProvider()
        current_price = market_data.get_current_price_sync(ticker, Config.EXCHANGE)
        if not current_price:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate ATM strikes
        atm_ce, atm_pe = self._calculate_atm_strikes(df, current_price)
        
        # Get strike ranges
        ce_strikes = self._get_strikes_range(df, atm_ce, df['strikePrice'].diff().median(), True)
        pe_strikes = self._get_strikes_range(df, atm_pe, df['strikePrice'].diff().median(), False)
        
        # Filter and process CE options
        ce_data = df[
            (df['strikePrice'].isin(ce_strikes)) & 
            (df['instrumentType'] == 'CE') &
            (df['expiryDate'].apply(self._format_expiry_date) == expiry_date)
        ].copy()
        
        # Filter and process PE options
        pe_data = df[
            (df['strikePrice'].isin(pe_strikes)) & 
            (df['instrumentType'] == 'PE') &
            (df['expiryDate'].apply(self._format_expiry_date) == expiry_date)
        ].copy()
        
        # Select required columns
        columns = ["strikePrice", "expiryDate", "lastPrice", "instrumentType"]
        ce_data = ce_data[columns].reset_index(drop=True)
        pe_data = pe_data[columns].reset_index(drop=True)
        
        return ce_data, pe_data

# Dashboard Component
class OptionChainDashboard:
    def __init__(self):
        self.stocks_data = self._load_stocks_data()
        self.processor = OptionChainProcessor()
    
    def _load_stocks_data(self) -> pd.DataFrame:
        """Load stocks data from CSV."""
        try:
            return pd.read_csv("FNO Stocks - All FO Stocks List, Technical Analysis Scanner.csv")
        except Exception as e:
            st.error(f"Error loading stocks data: {e}")
            return pd.DataFrame()
    
    def _get_stock_list(self, selected_option: str) -> List[str]:
        """Get stock list with selected option at the top."""
        if self.stocks_data.empty:
            return [selected_option]
        
        stock_list = self.stocks_data["Symbol"].tolist()
        if selected_option in stock_list:
            stock_list.remove(selected_option)
        
        return [selected_option] + stock_list
    
    def _create_controls(self, config: TableConfig) -> Tuple[str, str, int]:
        """Create UI controls for stock and expiry selection."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('##### Share List')
            selected_stock = st.selectbox(
                label="Select Stock",
                options=self._get_stock_list(config.selected_option),
                key=f"share_list{config.table_number}",
                label_visibility='collapsed'
            )
            
            # Get lot size
            lot_size = 1
            if not self.stocks_data.empty:
                stock_data = self.stocks_data[self.stocks_data["Symbol"] == selected_stock]
                if not stock_data.empty:
                    lot_size = stock_data['Jun-24'].iloc[0]
        
        with col2:
            st.markdown('##### Expiry List')
            expiry_date = st.selectbox(
                label="Select Expiry",
                options=DATE_LIST,
                key=f"exp_list{config.table_number}",
                label_visibility='collapsed'
            )
        
        return selected_stock, expiry_date, lot_size
    
    def _fetch_market_data(self, ticker: str) -> Tuple[float, float, float]:
        """Fetch market data for a ticker."""
        try:
            # Get current price
            market_data = MarketDataProvider()
            current_price = market_data.get_current_price_sync(ticker, Config.EXCHANGE)
            
            # Get 52-week range
            price_range = get_52_week_range(ticker, Config.EXCHANGE)
            
            return current_price or 0.0, price_range.low, price_range.high
            
        except Exception as e:
            st.error(f"Error fetching market data for {ticker}: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def _calculate_premium_matrix(self, ce_data: pd.DataFrame, pe_data: pd.DataFrame, 
                                current_price: float) -> pd.DataFrame:
        """Calculate premium percentage matrix."""
        if ce_data.empty or pe_data.empty or current_price <= 0:
            return pd.DataFrame(columns=["CE Premium%", "CE (Premium+SP)%", 
                                       "PE Premium%", "PE (Premium+SP)%"])
        
        # Calculate matrix size
        matrix_size = min(len(ce_data), len(pe_data))
        
        matrix_data = []
        for i in range(matrix_size):
            ce_price = ce_data.iloc[i]["lastPrice"]
            pe_price = pe_data.iloc[i]["lastPrice"]
            ce_strike = ce_data.iloc[i]["strikePrice"]
            pe_strike = pe_data.iloc[i]["strikePrice"]
            
            row = {
                "CE Premium%": round((ce_price / current_price) * 100, 2),
                "CE (Premium+SP)%": round(((ce_strike - current_price + ce_price) / current_price) * 100, 2),
                "PE Premium%": round((pe_price / current_price) * 100, 2),
                "PE (Premium+SP)%": round(((current_price - pe_strike + pe_price) / current_price) * 100, 2)
            }
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data)
    
    def _create_metrics_display(self, current_price: float, lot_size: int, 
                               low_52: float, high_52: float):
        """Display market metrics."""
        cols = st.columns(6)
        
        with cols[0]:
            st.markdown(f'##### CMP: {current_price:.2f}')
        with cols[1]:
            st.markdown(f'##### Lot Size: {lot_size}')
        with cols[2]:
            st.markdown(f'##### Contract Value: {lot_size * current_price:.2f}')
        with cols[3]:
            st.markdown(f'##### Time: {datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S")}')
        with cols[4]:
            st.markdown(f'##### 52 week low: {low_52:.2f}')
        with cols[5]:
            st.markdown(f'##### 52 week high: {high_52:.2f}')
    
    def _create_filters(self, df: pd.DataFrame, table_number: int) -> List[int]:
        """Create filter controls."""
        filters = st.columns(4)
        values = list(range(11))  # 0-10
        filter_values = []
        
        # CE filters
        with filters[1]:
            nested_cols = st.columns(2)
            for i, col in enumerate(['CE Premium%', 'CE (Premium+SP)%']):
                with nested_cols[i]:
                    filter_values.append(st.selectbox(
                        f'Filter {col}',
                        values,
                        key=f'filter_{table_number}_{i}'
                    ))
        
        # PE filters
        with filters[3]:
            nested_cols = st.columns(2)
            for i, col in enumerate(['PE Premium%', 'PE (Premium+SP)%']):
                with nested_cols[i]:
                    filter_values.append(st.selectbox(
                        f'Filter {col}',
                        values,
                        key=f'filter_{table_number}_{i+2}'
                    ))
        
        return filter_values
    
    def _display_ce_options_table(self, ce_data: pd.DataFrame, filtered_data: pd.DataFrame):
        """Display CE options table."""
        if not ce_data.empty and not filtered_data.empty:
            display_data = ce_data.loc[filtered_data.index].copy()
            display_data = display_data.rename(columns={
                'strikePrice': 'Strike Price',
                'expiryDate': 'Expiry Date',
                'lastPrice': 'Last Price',
                'instrumentType': 'Type'
            })
            
            styled_data = display_data.style.set_properties(
                **{'text-align': 'center', 'background-color': 'palegreen'}
            ).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            
            st.dataframe(styled_data)
        else:
            st.dataframe(pd.DataFrame(columns=['Strike Price', 'Expiry Date', 'Last Price', 'Type']))
    
    def _display_ce_premium_table(self, filtered_data: pd.DataFrame):
        """Display CE premium table."""
        if not filtered_data.empty:
            styled_data = filtered_data.style.applymap(
                lambda val: get_highlight_style(val, 'CE Premium%'), 
                subset=['CE Premium%']
            ).applymap(
                lambda val: get_highlight_style(val, 'CE (Premium+SP)%'), 
                subset=['CE (Premium+SP)%']
            ).set_properties(**{'text-align': 'center'})
            
            st.dataframe(styled_data)
        else:
            st.dataframe(pd.DataFrame(columns=['CE Premium%', 'CE (Premium+SP)%']))
    
    def _display_pe_options_table(self, pe_data: pd.DataFrame, filtered_data: pd.DataFrame):
        """Display PE options table."""
        if not pe_data.empty and not filtered_data.empty:
            display_data = pe_data.loc[filtered_data.index].copy()
            display_data = display_data.rename(columns={
                'strikePrice': 'Strike Price',
                'expiryDate': 'Expiry Date',
                'lastPrice': 'Last Price',
                'instrumentType': 'Type'
            })
            
            styled_data = display_data.style.set_properties(
                **{'text-align': 'center', 'background-color': 'antiquewhite'}
            ).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            
            st.dataframe(styled_data)
        else:
            st.dataframe(pd.DataFrame(columns=['Strike Price', 'Expiry Date', 'Last Price', 'Type']))
    
    def _display_pe_premium_table(self, filtered_data: pd.DataFrame):
        """Display PE premium table."""
        if not filtered_data.empty:
            styled_data = filtered_data.style.applymap(
                lambda val: get_highlight_style(val, 'PE Premium%'), 
                subset=['PE Premium%']
            ).applymap(
                lambda val: get_highlight_style(val, 'PE (Premium+SP)%'), 
                subset=['PE (Premium+SP)%']
            ).set_properties(**{'text-align': 'center'})
            
            st.dataframe(styled_data)
        else:
            st.dataframe(pd.DataFrame(columns=['PE Premium%', 'PE (Premium+SP)%']))
    
    def _create_data_tables(self, ce_data: pd.DataFrame, pe_data: pd.DataFrame,
                           matrix_df: pd.DataFrame, filter_values: List[int]):
        """Create the four data tables."""
        cols = st.columns(4)
        
        # Apply filters
        ce_filtered = matrix_df[
            (matrix_df['CE Premium%'] >= filter_values[0]) & 
            (matrix_df['CE (Premium+SP)%'] >= filter_values[1])
        ]
        pe_filtered = matrix_df[
            (matrix_df['PE Premium%'] >= filter_values[2]) & 
            (matrix_df['PE (Premium+SP)%'] >= filter_values[3])
        ]
        
        # CE Options Table
        with cols[0]:
            self._display_ce_options_table(ce_data, ce_filtered)
        
        # CE Premium Table
        with cols[1]:
            self._display_ce_premium_table(ce_filtered)
        
        # PE Options Table
        with cols[2]:
            self._display_pe_options_table(pe_data, pe_filtered)
        
        # PE Premium Table
        with cols[3]:
            self._display_pe_premium_table(pe_filtered)
    
    def render_table(self, config: TableConfig):
        """Main method to render a complete option chain table."""
        st.write("---")
        
        # Create controls
        selected_stock, expiry_date, lot_size = self._create_controls(config)
        
        # Fetch data
        ce_data, pe_data = self.processor.get_option_chain(selected_stock, expiry_date)
        current_price, low_52, high_52 = self._fetch_market_data(selected_stock)
        
        # Calculate matrix
        matrix_df = self._calculate_premium_matrix(ce_data, pe_data, current_price)
        
        # Display metrics
        self._create_metrics_display(current_price, lot_size, low_52, high_52)
        
        # Create filters
        filter_values = self._create_filters(matrix_df, config.table_number)
        
        # Display tables
        self._create_data_tables(ce_data, pe_data, matrix_df, filter_values)

# History Manager
class HistoryManager:
    def __init__(self, history_file: str = "history.csv"):
        self.history_file = history_file
        self.history_data = self._load_history()
    
    def _load_history(self) -> pd.DataFrame:
        """Load history data with error handling."""
        try:
            return pd.read_csv(self.history_file)
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return pd.DataFrame()
    
    def get_last_configuration(self) -> List[TableConfig]:
        """Get last configuration from history."""
        if len(self.history_data) <= 1:
            return self._get_default_configs()
        
        try:
            last_record = self.history_data.tail(1).iloc[0]
            
            configs = []
            for i in range(1, 4):
                table_key = f'table{i}'
                exp_key = f'exp{i}'
                
                if table_key in last_record and exp_key in last_record:
                    stock = last_record[table_key]
                    expiry = last_record[exp_key]
                    
                    # Validate expiry date
                    if expiry not in DATE_LIST:
                        expiry = EXP_OPTION
                    
                    configs.append(TableConfig(i, stock, expiry))
                else:
                    configs.append(self._get_default_config(i))
            
            return configs
            
        except Exception as e:
            logger.error(f"Error processing history: {e}")
            return self._get_default_configs()
    
    def _get_default_configs(self) -> List[TableConfig]:
        """Get default configurations."""
        default_stocks = ['RELIANCE', 'VEDL', 'INFY']
        return [
            TableConfig(i+1, stock, EXP_OPTION)
            for i, stock in enumerate(default_stocks)
        ]
    
    def _get_default_config(self, table_number: int) -> TableConfig:
        """Get default configuration for a specific table."""
        default_stocks = ['RELIANCE', 'VEDL', 'INFY']
        return TableConfig(table_number, default_stocks[table_number-1], EXP_OPTION)
    
    def save_configuration(self, configs: List[TableConfig]):
        """Save current configuration to history."""
        try:
            data = {
                'table1': [configs[0].selected_option],
                'exp1': [configs[0].exp_option],
                'table2': [configs[1].selected_option],
                'exp2': [configs[1].exp_option],
                'table3': [configs[2].selected_option],
                'exp3': [configs[2].exp_option],
                'timestamp': [datetime.datetime.now()]
            }
            
            new_record = pd.DataFrame(data)
            
            if len(self.history_data) > 30:
                new_record.to_csv(self.history_file, index=False, header=True)
            else:
                new_record.to_csv(self.history_file, mode='a', index=False, header=False)
                
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")

# Main application
def main():
    """Main application entry point."""
    st.markdown('## LIVE OPTION CHAIN ANALYSIS (OPTSTK)')
    
    # Initialize components
    history_manager = HistoryManager()
    dashboard = OptionChainDashboard()
    
    # Get configurations
    configs = history_manager.get_last_configuration()
    
    # Render tables
    for config in configs:
        dashboard.render_table(config)
    
    # Save current configuration
    history_manager.save_configuration(configs)

if __name__ == "__main__":
    main()
