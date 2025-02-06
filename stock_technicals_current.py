import requests
import json
from datetime import datetime, timedelta, date
import time
import os
import calendar
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_month_dates(year, month):
    """Helper function to get start and end dates for a given month"""
    start_date = datetime(year, month, 1)
    _, last_day = calendar.monthrange(year, month)
    end_date = datetime(year, month, last_day, 23, 59, 59)
    
    # If it's the current month, use today as end date
    current_date = datetime.now()
    if year == current_date.year and month == current_date.month:
        end_date = current_date
        
    return start_date, end_date

def get_economic_events(api_key, output_dir):
    """Fetch economic events and indicators"""
    events = {
        'GDP': 'REAL_GDP',
        'FEDERAL_FUNDS_RATE': 'FEDERAL_FUNDS_RATE',
        'TREASURY_YIELD': 'TREASURY_YIELD',
        'CPI': 'CPI',
        'INFLATION': 'INFLATION',
        'UNEMPLOYMENT': 'UNEMPLOYMENT',
        'NONFARM_PAYROLL': 'NONFARM_PAYROLL',
        'RETAIL_SALES': 'RETAIL_SALES',
        'DURABLES': 'DURABLES'
    }
    
    for event_name, function in events.items():
        time.sleep(1)  # Rate limiting
        url = f'https://www.alphavantage.co/query?function={function}&apikey={api_key}'
        
        try:
            response = requests.get(url)
            data = response.json()
            
            output_file = os.path.join(output_dir, f"{event_name.lower()}.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Saved {event_name} data to {output_file}")
            
        except Exception as e:
            print(f"Error fetching {event_name}: {str(e)}")

def get_earnings_calendar(api_key, output_dir):
    """Fetch earnings calendar for major companies"""
    url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        output_file = os.path.join(output_dir, "earnings_calendar.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Saved earnings calendar to {output_file}")
        
    except Exception as e:
        print(f"Error fetching earnings calendar: {str(e)}")

def get_ipo_calendar(api_key, output_dir):
    """Fetch IPO calendar"""
    url = f'https://www.alphavantage.co/query?function=IPO_CALENDAR&apikey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        output_file = os.path.join(output_dir, "ipo_calendar.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Saved IPO calendar to {output_file}")
        
    except Exception as e:
        print(f"Error fetching IPO calendar: {str(e)}")

def get_intraday_price_history(symbol, api_key, interval='5min', start_date=None, end_date=None):
    """Helper function to get intraday price history for a specific date range"""
    # For recent data (current and previous month), use TIME_SERIES_INTRADAY
    if start_date >= (datetime.now() - timedelta(days=60)):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=full&apikey={api_key}&entitlement=delayed&month={start_date.strftime("%Y-%m")}'
    else:
        # For older data, use TIME_SERIES_INTRADAY with month parameter and extended=true
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=full&apikey={api_key}&month={start_date.strftime("%Y-%m")}&extended=true'
    
    r = requests.get(url)
    data = r.json()
    
    try:
        if f"Time Series ({interval})" not in data:
            if "Note" in data:
                print(f"API Note: {data['Note']}")
                time.sleep(60)  # Wait for API rate limit to reset if needed
                # Retry the request
                r = requests.get(url)
                data = r.json()
                if f"Time Series ({interval})" not in data:
                    raise Exception(f"Error fetching intraday price data: {data}")
            else:
                raise Exception(f"Error fetching intraday price data: {data}")
        
        time_series_data = data[f"Time Series ({interval})"]
        
        # Convert JSON format to list of dictionaries
        prices = []
        for timestamp, values in time_series_data.items():
            try:
                if len(timestamp) == 16:
                    timestamp = timestamp + ":00"
                
                candle_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                
                if start_date <= candle_time <= end_date:
                    prices.append({
                        "timestamp": timestamp,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"])
                    })
            except ValueError:
                continue
                
        if not prices:
            print(f"No data points found for {start_date.strftime('%Y-%m')}")
            
        return sorted(prices, key=lambda x: x["timestamp"])
    except Exception as e:
        print(f"Error processing price data: {str(e)}")
        return []

def get_technical_indicator(symbol, api_key, indicator, output_file, interval='5min', start_date=None, end_date=None, **params):
    """Generic function to fetch technical indicators from Alpha Vantage"""
    base_url = 'https://www.alphavantage.co/query'
    
    # For recent data (current and previous month), use regular endpoint
    if start_date >= (datetime.now() - timedelta(days=60)):
        query_params = {
            'function': f'{indicator}',
            'symbol': symbol,
            'interval': interval,
            'apikey': api_key,
            'outputsize': 'full',
            'month': start_date.strftime("%Y-%m"),
            **params
        }
    else:
        # For older data, use extended parameter
        query_params = {
            'function': f'{indicator}',
            'symbol': symbol,
            'interval': interval,
            'apikey': api_key,
            'outputsize': 'full',
            'month': start_date.strftime("%Y-%m"),
            'extended': 'true',
            **params
        }
    
    try:
        response = requests.get(base_url, params=query_params)
        data = response.json()
        
        # Check for API rate limit
        if "Note" in data:
            print(f"API Note: {data['Note']}")
            time.sleep(60)  # Wait for API rate limit to reset
            response = requests.get(base_url, params=query_params)
            data = response.json()
        
        # Different indicators have different response formats
        indicator_data_key = ""
        if indicator == "RSI":
            indicator_data_key = "Technical Analysis: RSI"
        elif indicator == "MACD":
            indicator_data_key = "Technical Analysis: MACD"
        elif indicator == "BBANDS":
            indicator_data_key = "Technical Analysis: BBANDS"
        elif indicator == "STOCH":
            indicator_data_key = "Technical Analysis: STOCH"
        else:
            indicator_data_key = f"Technical Analysis: {indicator}"
            
        if indicator_data_key not in data:
            raise Exception(f"Error fetching {indicator} data: {data.get('Note', 'Unknown error')}")
            
        formatted_data = {
            "symbol": symbol,
            "indicator": indicator,
            "parameters": params,
            "interval": interval,
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "data": []
        }
        
        # Get all timestamps from the indicator data within our time window
        for timestamp, values in data[indicator_data_key].items():
            try:
                if len(timestamp) == 16:
                    timestamp = timestamp + ":00"
                    
                data_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                
                # Only include data points within our time window
                if start_date <= data_time <= end_date:
                    entry = {"timestamp": timestamp}
                    entry.update({k: float(v) for k, v in values.items()})
                    formatted_data["data"].append(entry)
            except ValueError:
                continue
            
        formatted_data["data"].sort(key=lambda x: x["timestamp"])
        
        if not formatted_data["data"]:
            raise Exception(f"No valid data points found for {indicator} within the specified time range")
            
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"{indicator} data saved to {output_file} with {len(formatted_data['data'])} data points")
        
    except Exception as e:
        print(f"Error calculating {indicator}: {str(e)}")
        if "Note" in str(e):
            print("Waiting 60 seconds for API rate limit reset...")
            time.sleep(60)

def get_monthly_data(symbol, api_key, base_dir, interval='5min', months_back=12):
    """Fetches and organizes data by month for the specified number of past months"""
    # Create base directory for the stock and economic events
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    events_dir = "Economic_EVENTS"
    if not os.path.exists(events_dir):
        os.makedirs(events_dir)
        print("\nFetching economic events and market calendars...")
        get_economic_events(api_key, events_dir)
        get_earnings_calendar(api_key, events_dir)
        get_ipo_calendar(api_key, events_dir)
    
    # Get current date info
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Define indicator configurations
    indicators = {
        # Moving Averages
        'MA_TYPES': [
            ('SMA', {'time_period': 20, 'series_type': 'close'}),
            ('EMA', {'time_period': 20, 'series_type': 'close'}),
            ('WMA', {'time_period': 20, 'series_type': 'close'}),
            ('DEMA', {'time_period': 20, 'series_type': 'close'}),
            ('TEMA', {'time_period': 20, 'series_type': 'close'}),
            ('TRIMA', {'time_period': 20, 'series_type': 'close'}),
            ('KAMA', {'time_period': 20, 'series_type': 'close'}),
            ('MAMA', {'series_type': 'close'}),
            ('T3', {'time_period': 20, 'series_type': 'close'}),
            ('VWAP', {})
        ],
        # Momentum Indicators
        'MOMENTUM': [
            ('MACD', {'series_type': 'close', 'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            ('RSI', {'time_period': 14, 'series_type': 'close'}),
            ('STOCH', {}),
            ('STOCHF', {}),
            ('STOCHRSI', {'time_period': 14, 'series_type': 'close'}),
            ('ADX', {'time_period': 14}),
            ('ADXR', {'time_period': 14}),
            ('APO', {'series_type': 'close', 'fastperiod': 12, 'slowperiod': 26}),
            ('PPO', {'series_type': 'close', 'fastperiod': 12, 'slowperiod': 26}),
            ('MOM', {'time_period': 10, 'series_type': 'close'}),
            ('BOP', {}),
            ('CCI', {'time_period': 20}),
            ('CMO', {'time_period': 14, 'series_type': 'close'}),
            ('ROC', {'time_period': 10, 'series_type': 'close'}),
            ('ROCR', {'time_period': 10, 'series_type': 'close'}),
            ('AROON', {'time_period': 14}),
            ('AROONOSC', {'time_period': 14}),
            ('MFI', {'time_period': 14}),
            ('TRIX', {'time_period': 30, 'series_type': 'close'}),
            ('ULTOSC', {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28}),
            ('DX', {'time_period': 14}),
            ('MINUS_DI', {'time_period': 14}),
            ('PLUS_DI', {'time_period': 14}),
            ('MINUS_DM', {'time_period': 14}),
            ('PLUS_DM', {'time_period': 14}),
            ('WILLR', {'time_period': 14})
        ],
        # Volatility Indicators
        'VOLATILITY': [
            ('ATR', {'time_period': 14}),
            ('NATR', {'time_period': 14}),
            ('TRANGE', {}),
            ('BBANDS', {'time_period': 20, 'series_type': 'close', 'nbdevup': 2, 'nbdevdn': 2}),
            ('MIDPOINT', {'time_period': 14, 'series_type': 'close'}),
            ('MIDPRICE', {'time_period': 14})
        ],
        # Volume Indicators
        'VOLUME': [
            ('OBV', {}),
            ('AD', {}),
            ('ADOSC', {'fastperiod': 3, 'slowperiod': 10}),
            ('HT_DCPERIOD', {'series_type': 'close'}),
            ('HT_DCPHASE', {'series_type': 'close'}),
            ('HT_PHASOR', {'series_type': 'close'}),
            ('HT_SINE', {'series_type': 'close'}),
            ('HT_TRENDMODE', {'series_type': 'close'})
        ]
    }
    
    # Process each month for the specified number of months_back
    for i in range(months_back):
        # Calculate year and month accounting for wrap-around
        if current_month - i <= 0:
            year = current_year - 1
            month = 12 + (current_month - i)
        else:
            year = current_year
            month = current_month - i
            
        # Create month directory
        month_dir = os.path.join(base_dir, f"{year}_{month:02d}")
        if not os.path.exists(month_dir):
            os.makedirs(month_dir)
            
        # Get start and end dates for the month (full month)
        start_date, end_date = get_month_dates(year, month)
        
        print(f"\nProcessing data for {calendar.month_name[month]} {year}...")
        print(f"Time window: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Get price history for the month
            prices = get_intraday_price_history(symbol, api_key, interval, start_date, end_date)
            
            if prices:
                formatted_data = {
                    "symbol": symbol,
                    "period_start": start_date.strftime("%Y-%m-%d"),
                    "period_end": end_date.strftime("%Y-%m-%d"),
                    "interval": interval,
                    "candles": prices
                }
                
                # Save price data
                price_file = os.path.join(month_dir, f"{symbol}_DATA.json")
                with open(price_file, 'w') as f:
                    json.dump(formatted_data, f, indent=4)
                print(f"Price data saved to {price_file} with {len(prices)} data points")
                
                # Calculate technical indicators
                for category in ['MA_TYPES', 'MOMENTUM', 'VOLATILITY', 'VOLUME']:
                    for indicator_config in indicators[category]:
                        time.sleep(1)  # Rate limiting
                        indicator_name, params = indicator_config
                        get_technical_indicator(
                            symbol,
                            api_key,
                            indicator_name,
                            os.path.join(month_dir, f"{indicator_name.lower()}.json"),
                            interval=interval,
                            start_date=start_date,
                            end_date=end_date,
                            **params
                        )
            else:
                print(f"No price data available for {calendar.month_name[month]} {year}")
                
        except Exception as e:
            print(f"Error processing {calendar.month_name[month]} {year}: {str(e)}")
            continue

# Example usage:
if __name__ == "__main__":
    api_key = os.getenv('ALPHA_VANTAGE_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_KEY not found in environment variables")
    ticker = input("Enter the stock ticker: ").upper().strip()
    folder_name = f"{ticker}_HISTORICAL"
    try:
        months = int(input("How many months back do you want to extract the data for? Input a single number: ").strip())
    except ValueError:
        print("Invalid input. Defaulting to 12 months.")
        months = 12
    get_monthly_data(ticker, api_key, folder_name, interval='5min', months_back=months)