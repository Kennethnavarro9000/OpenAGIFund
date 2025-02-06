from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import argparse

#reasoning model prompt in trello
# use suggested reasoning traces prompt for non reasoning models
load_dotenv()

# Base template for the analysis
ANALYSIS_TEMPLATE = """Assess whether {technical_indicator} provides true alpha for {stock_ticker} between {start_date} and {end_date}.
> 1. Compare returns against relevant market benchmarks to isolate alpha.
> 2. Account for the following critical external market dates: {market_dates}, examining their influence on stock performance and indicator signals.
> 3. Evaluate how the indicator behaves both overall and in periods unaffected by these broader market forces.
> 4. Summarize the indicator's predictive validity, citing specific statistical or trend-based evidence to confirm or refute alpha generation."""

def get_user_inputs() -> Dict[str, str]:
    """Get required inputs from user in natural language format"""
    print("\nPlease provide the following information:")
    
    ticker = input("What stock ticker would you like to analyze? ").strip().upper()
    indicator = input("Which technical indicator would you like to evaluate? ").strip()
    date_range = input("What is the time period for analysis? (e.g., 'from January 2023 to December 2023') ").strip()
    
    # Parse the date range
    try:
        # This is a simple parse - you might want to use a more sophisticated date parser
        start_str, end_str = date_range.lower().replace('from', '').replace('to', '').split('and')
        start_date = pd.to_datetime(start_str.strip()).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_str.strip()).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing dates: {e}")
        print("Using default date format...")
        start_date = input("Please enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Please enter end date (YYYY-MM-DD): ").strip()
    
    return {
        "stock_ticker": ticker,
        "technical_indicator": indicator,
        "start_date": start_date,
        "end_date": end_date
    }

def load_market_dates() -> List[Dict[str, Any]]:
    """Load and structure market-moving dates from the file"""
    try:
        with open('market_moving_dates.txt', 'r') as f:
            lines = f.readlines()[2:]  # Skip header lines
            dates = []
            for line in lines:
                if '|' in line:
                    date, direction, confidence = [x.strip() for x in line.split('|')]
                    dates.append({
                        "date": date,
                        "direction": direction,
                        "confidence": float(confidence)
                    })
            return sorted(dates, key=lambda x: x["date"])
    except Exception as e:
        print(f"Error loading market dates: {e}")
        return []

def format_market_dates(dates: List[Dict[str, Any]]) -> str:
    """Format market dates into a concise string for the prompt"""
    formatted_dates = []
    for date in dates:
        formatted_dates.append(
            f"{date['date']} ({date['direction']}, confidence: {date['confidence']:.2f})"
        )
    return "[" + ", ".join(formatted_dates) + "]"

def parse_data_with_llm(content: str, file_path: str) -> Dict[str, Any]:
    """Use LLM to intelligently parse and format data"""
    llm = ChatOpenAI(model="o1-mini")
    
    # Sample of the content for analysis
    content_sample = content[:1000] if len(content) > 1000 else content
    
    analysis = llm.invoke(
        f"""Analyze and parse this data content from {file_path}:

        {content_sample}

        Identify:
        1. The data format and structure
        2. Any headers or metadata (like indicator type, parameters)
        3. The time series data format

        Convert this into a structured format with:
        - metadata (if any)
        - values (as date-value pairs)

        Respond in valid JSON format like this example:
        {{
            "metadata": {{
                "indicator_type": "type if found",
                "parameters": "params if found"
            }},
            "values": {{
                "YYYY-MM-DD": numeric_value,
                ...
            }}
        }}
        """
    )
    
    try:
        # Extract the JSON part from the response
        response_text = analysis.content
        # Find the JSON block (between first { and last })
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in LLM response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")

def load_data(file_path: str) -> Dict[str, Any]:
    """Load data from any format using LLM for intelligent parsing"""
    try:
        # First try standard JSON loading
        if file_path.lower().endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return data
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to LLM
                pass
        
        # For all other cases or failed JSON, use LLM parsing
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Use LLM to parse the content
        print(f"\nUsing AI to parse data format from {file_path}...")
        parsed_data = parse_data_with_llm(content, file_path)
        print("Data parsed successfully!")
        return parsed_data
        
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")

def validate_data_paths(stock_path: str, indicator_path: str) -> Tuple[bool, str, Dict[str, Any], Dict[str, Any]]:
    """Validate that the data files exist and are readable, and return their contents"""
    stock_file = Path(stock_path)
    indicator_file = Path(indicator_path)
    
    if not stock_file.exists():
        return False, f"Stock data file not found: {stock_path}", None, None
    if not indicator_file.exists():
        return False, f"Indicator data file not found: {indicator_path}", None, None
    
    try:
        # Stock data should still be JSON
        stock_data = load_data(stock_path)
        # Indicator data can be either JSON or TXT
        indicator_data = load_data(indicator_path)
        
        return True, "Data files validated successfully", stock_data, indicator_data
    except ValueError as e:
        return False, str(e), None, None

def format_data_preview(data: Dict[str, Any], is_json: bool = True) -> str:
    """Format data preview based on file type"""
    if not data:
        return "No data available"
    
    try:
        preview = []
        
        # Handle metadata if present
        if isinstance(data, dict) and 'metadata' in data:
            metadata = data['metadata']
            for key, value in metadata.items():
                if value:  # Only show non-empty metadata
                    preview.append(f"{key.title()}: {value}")
            preview.append("")  # Empty line for readability
        
        # Handle values
        values = data.get('values', data)  # Use 'values' if present, otherwise use whole dict
        if isinstance(values, dict):
            sorted_items = sorted(values.items())[:5]
            for date, value in sorted_items:
                if isinstance(value, (int, float)):
                    preview.append(f"Date: {date}, Value: {value:.6f}")
                else:
                    preview.append(f"Date: {date}, Value: {value}")
        
        return "\n".join(preview)
    except Exception as e:
        return f"Error formatting preview: {e}"

def create_analysis_prompt(user_inputs: Dict[str, str], market_dates: List[Dict[str, Any]], 
                         stock_path: str, indicator_path: str) -> Dict[str, Any]:
    """Create the final analysis prompt combining all components"""
    
    # Validate data paths and load data
    is_valid, message, stock_data, indicator_data = validate_data_paths(stock_path, indicator_path)
    if not is_valid:
        raise ValueError(message)
    
    # Format market dates
    formatted_dates = format_market_dates(market_dates)
    
    # Create the prompt
    prompt = ANALYSIS_TEMPLATE.format(
        technical_indicator=user_inputs["technical_indicator"],
        stock_ticker=user_inputs["stock_ticker"],
        start_date=user_inputs["start_date"],
        end_date=user_inputs["end_date"],
        market_dates=formatted_dates
    )
    
    # Create the complete context
    context = {
        "prompt": prompt,
        "metadata": {
            "user_inputs": user_inputs,
            "market_dates": market_dates,
            "data_paths": {
                "stock_data": stock_path,
                "indicator_data": indicator_path
            }
        },
        "data": {
            "stock_data": stock_data,
            "indicator_data": indicator_data
        }
    }
    
    return context

def main():
    # Get user inputs
    user_inputs = get_user_inputs()
    
    # Load market dates
    market_dates = load_market_dates()
    if not market_dates:
        print("No market dates found. Please run date_organizer.py first.")
        return
    
    # Get data file paths
    print("\nPlease provide the paths to your data files:")
    print("Note: Stock data should be in JSON format")
    print("Technical indicator data can be in either JSON or TXT format")
    stock_path = input("Path to stock price data file (e.g., 'AMD_DATA/stock_prices.json'): ").strip()
    indicator_path = input(f"Path to {user_inputs['technical_indicator']} data file (e.g., 'AMD_DATA/SMA_daily.json' or 'AMD_DATA/SMA_daily.txt'): ").strip()
    
    try:
        # Create the analysis prompt
        context = create_analysis_prompt(user_inputs, market_dates, stock_path, indicator_path)
        
        # Save the results
        with open('analysis_prompt.json', 'w') as f:
            json.dump(context, f, indent=2)
        
        # Save human-readable version
        with open('analysis_prompt.txt', 'w') as f:
            f.write("Technical Analysis Prompt\n")
            f.write("=" * 50 + "\n\n")
            f.write(context["prompt"])
            f.write("\n\n")
            f.write("Data Sources:\n")
            f.write(f"Stock Data: {stock_path}\n")
            f.write(f"Indicator Data: {indicator_path}\n")
            f.write("\nData Preview:\n")
            f.write("-" * 30 + "\n")
            f.write("Stock Data Sample:\n")
            f.write(format_data_preview(context["data"]["stock_data"], is_json=True))
            f.write("\n\nIndicator Data Sample:\n")
            is_json_indicator = Path(indicator_path).suffix.lower() == '.json'
            f.write(format_data_preview(context["data"]["indicator_data"], is_json=is_json_indicator))
        
        print("\nPrompt generation complete! Results saved to:")
        print("1. analysis_prompt.json (Structured data for programmatic use)")
        print("2. analysis_prompt.txt (Human-readable format)")
        print("\nData has been validated and included in the context")
        
    except Exception as e:
        print(f"Error generating prompt: {e}")

if __name__ == "__main__":
    main() 