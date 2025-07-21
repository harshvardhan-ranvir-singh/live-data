Live Option Chain Analysis System

This is a Streamlit-based web application for real-time analysis of Indian stock options (F&O - Futures and Options).
A real-time options chain analysis dashboard using Python and Streamlit that provides live market data visualization for Indian F&O (Futures & Options) markets. The application integrates multiple financial APIs (NSE India, Google Finance) to fetch option chain data, current market prices, and 52-week high/low values. Built an interactive web interface with dynamic filtering capabilities, premium percentage calculations, and color-coded risk indicators for both call and put options. Implemented session persistence, responsive design, and multi-table analysis allowing traders to simultaneously monitor up to three different stocks with customizable expiry dates. The system processes real-time option chain data for 183+ F&O stocks, calculating key metrics like premium percentages and strike price differentials to assist in options trading decisions.
Here's a complete breakdown.


System Architecture
Main Application: experiment.py - A Streamlit web app that provides live option chain analysis for NSE (National Stock Exchange) stocks.

Core Functionality

1. Option Chain Data Retrieval
Data Source: NSE India API (https://www.nseindia.com/api/option-chain-equities)
Real-time Price: Google Finance API for current market prices
52-week High/Low: Scraped from Google Finance

2. Key Features
📈 Live Market Data
Real-time option chain data for any F&O stock
Current market price (CMP) tracking
52-week high/low analysis
Lot size and contract value calculations

🎯 Option Analysis Matrix
The system calculates four key metrics for both Call (CE) and Put (PE) options:
  1. CE Premium%: Call option premium as percentage of stock price
  2. CE (Premium+SP)%: Call premium + strike price difference as percentage
  3.PE Premium%: Put option premium as percentage of stock price
  4. PE (Premium+SP)%: Put premium + strike price difference as percentage

🔍 Filtering System
Interactive filters (0-10 range) for each metric
Color-coded highlighting (paleturquoise for >1%, wheat for >5%)
Separate tables for CE and PE options
****************************************************************
📁 Data Files

FNO Stocks - All FO Stocks List, Technical Analysis Scanner.csv

  Contains 183 F&O stocks with their lot sizes
  Used as dropdown for stock selection

history.csv
  Tracks user selections across sessions
  Stores last 3 table configurations
  Maintains history for persistence

lot_size.csv
  Duplicate lot size data (appears redundant)
*************************************************************************************
User Interface

Multi-Table Dashboard
3 independent analysis tables (fragments)
Each table has:
Stock selector dropdown
Expiry date selector
Live market data display

4-column layout showing:
CE options data (green background)
CE analysis matrix (with highlighting)
PE options data (antique white background)
PE analysis matrix (with highlighting)

Real-time Updates:
Live timestamp display
Responsive design for mobile/desktop
Auto-refresh capabilities
**************************************************
🔧 Technical Implementation
Key Functions:
last_thursdays(): Calculates monthly expiry dates
current_market_price(): Real-time price generator
fifty_two_week_high_low(): Historical price analysis
get_dataframe(): Option chain data processing
highlight_ratio(): Conditional formatting
Data Processing:
ATM (At-The-Money) strike calculation
5 strikes above/below ATM for analysis
Premium percentage calculations
Strike price adjustments
****************************************************
💡 Use Case
This is a professional trading tool designed for:
Options traders analyzing premium structures
Risk assessment of option positions
Real-time market monitoring during trading hours
Comparative analysis across multiple stocks

*****************************************************
🚀 How to Run

Run: streamlit run experiment.py

The application will open in a web browser with live option chain analysis capabilities for Indian F&O markets.
This is essentially a sophisticated options analysis dashboard that helps traders make informed decisions by providing real-time data visualization and analysis tools for Indian equity options.
