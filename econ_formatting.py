from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('economic_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

load_dotenv()

@dataclass
class WeirdEvent:
    date: str
    description: str
    data_type: str
    value: float
    source_file: str
    severity: float = 0.0
    category: str = ""
    potential_market_impact: str = ""

    @classmethod
    def from_llm_response(cls, line: str, data_name: str, source_file: str, category_mapping: Dict[str, str]) -> Optional['WeirdEvent']:
        """Safely parse LLM response line into WeirdEvent object"""
        try:
            parts = line.replace('POINT:', '').strip().split('|')
            if len(parts) != 5:
                logging.warning(f"Invalid number of parts in line: {line}")
                return None
                
            # Parse and validate date
            date = parts[0].strip()
            if not date:
                logging.warning(f"Empty date in line: {line}")
                return None
                
            # Parse and validate value with percentage handling
            try:
                value_str = parts[1].strip()
                if value_str.endswith('%'):
                    value = float(value_str.rstrip('%')) / 100.0
                else:
                    value = float(value_str)
            except ValueError:
                logging.warning(f"Invalid value in line: {line}")
                return None
                
            # Parse and validate severity
            try:
                severity = float(parts[2].strip())
                if not 0 <= severity <= 1:
                    logging.warning(f"Severity out of range [0,1] in line: {line}")
                    severity = max(0, min(1, severity))  # Clamp to valid range
            except ValueError:
                logging.warning(f"Invalid severity in line: {line}")
                return None
                
            description = parts[3].strip()
            market_impact = parts[4].strip()
            
            if not description or not market_impact:
                logging.warning(f"Missing description or market impact in line: {line}")
                return None
                
            category = category_mapping.get(data_name, "other")
            
            return cls(
                date=date,
                description=description,
                data_type=data_name,
                value=value,
                source_file=source_file,
                severity=severity,
                category=category,
                potential_market_impact=market_impact
            )
        except Exception as e:
            logging.error(f"Error parsing line: {line}", exc_info=True)
            return None

class FileAnalyzer:
    def __init__(self, filename: str):
        self.filename = filename
        self.llm = ChatOpenAI(model="o1-mini")
        
        self.category_mapping = {
            "Federal Funds Rate": "monetary_policy",
            "Treasury Yield": "interest_rates",
            "Unemployment Rate": "employment",
            "Inflation Rate": "inflation",
            "Retail Sales": "consumer_spending",
            "Nonfarm Payroll": "employment",
            "GDP": "economic_growth",
            "Durable Goods Orders": "manufacturing",
            "CPI": "inflation"
        }
    
    def analyze(self) -> List[WeirdEvent]:
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON file: {self.filename}", exc_info=True)
            return []
        except FileNotFoundError:
            logging.error(f"File not found: {self.filename}")
            return []
            
        # Get the unit and determine if it's a percentage
        unit = data.get('unit', 'Unknown')
        is_percentage = unit.lower() in ['percent', 'percentage', '%']
        
        # Format data points with proper handling of percentages
        data_points = []
        for point in data.get("data", []):
            value = point.get('value')
            if is_percentage:
                data_points.append(f"Date: {point['date']}, Value: {value}%")
            else:
                data_points.append(f"Date: {point['date']}, Value: {value}")
        
        if not data_points:
            logging.warning(f"No data points found in {self.filename}")
            return []
        
        try:
            # Enhanced prompt with clear instructions about percentage handling
            analysis = self.llm.invoke(
                f"""Analyze this time series data for {data.get('name', 'Unknown')} ({unit}) and identify unusual points.
                
                Data Points:
                {chr(10).join(data_points)}
                
                For each unusual point, provide:
                1. The specific date and value (if the value is a percentage, include the % symbol)
                2. A detailed description of why it's unusual
                3. A severity rating (0.0-1.0) indicating how unusual/significant this event is
                4. Potential impact on financial markets
                
                Respond in EXACTLY this format (one per line):
                POINT: [DATE] | [VALUE] | [SEVERITY] | [DESCRIPTION] | [MARKET_IMPACT]
                
                Examples:
                For percentage values:
                POINT: 2024-01-01 | 5.2% | 0.8 | Largest monthly increase in 5 years | Market likely to react negatively
                
                For non-percentage values:
                POINT: 2024-01-01 | 1500.5 | 0.7 | Unexpected surge in volume | Positive market sentiment expected
                """
            )
        except Exception as e:
            logging.error(f"LLM analysis failed for {self.filename}", exc_info=True)
            return []
        
        weird_events = []
        for line in analysis.content.split('\n'):
            if line.strip().startswith('POINT:'):
                event = WeirdEvent.from_llm_response(
                    line, 
                    data.get('name', 'Unknown'), 
                    self.filename,
                    self.category_mapping
                )
                if event:
                    weird_events.append(event)
                    logging.info(f"Successfully parsed event from {self.filename}: {event.date}")
                
        if not weird_events:
            logging.warning(f"No valid events parsed from {self.filename}")
            logging.debug(f"LLM response was: {analysis.content}")
            
        return weird_events

class ResultMerger:
    def __init__(self):
        self.llm = ChatOpenAI(model="o1-mini")
        
    def analyze_market_impact(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze events for significant market-wide impact"""
        major_market_events = []
        
        for event in events:
            # Combine all market impact descriptions for multi-event dates
            market_impacts = [e['market_impact'] for e in event['events']]
            combined_impacts = "\n".join(market_impacts)
            
            # Ask LLM to analyze if this is a major market-moving event
            analysis = self.llm.invoke(
                f"""For the following economic event(s) on {event['date']} with overall severity {event['overall_severity']}, 
                analyze if this represents a major market-moving event that likely affected the entire market significantly.
                
                Event Summary: {event['summary']}
                Market Impacts:
                {combined_impacts}
                
                Consider:
                1. Did this event likely cause broad market movement (affecting most sectors)?
                2. Was the market impact strong and clear?
                3. What was the likely direction (positive/negative)?
                
                Respond in this format:
                MAJOR_MARKET_EVENT: [YES/NO]
                DIRECTION: [POSITIVE/NEGATIVE/MIXED]
                CONFIDENCE: [0-1]
                EXPLANATION: [Brief explanation]
                """
            )
            
            # Parse the response
            try:
                lines = analysis.content.strip().split('\n')
                is_major = lines[0].split(': ')[1].strip().upper() == 'YES'
                direction = lines[1].split(': ')[1].strip()
                confidence = float(lines[2].split(': ')[1].strip())
                explanation = lines[3].split(': ')[1].strip()
                
                if is_major and confidence >= 0.7:  # Only include high-confidence major events
                    major_market_events.append({
                        "date": event['date'],
                        "events": event['events'],
                        "overall_severity": event['overall_severity'],
                        "market_direction": direction,
                        "confidence": confidence,
                        "explanation": explanation,
                        "original_summary": event['summary']
                    })
            except (IndexError, ValueError) as e:
                logging.warning(f"Failed to parse market impact analysis for {event['date']}: {e}")
                continue
                
        return major_market_events
        
    def merge_results(self, all_events: List[List[WeirdEvent]]) -> List[Dict[str, Any]]:
        flat_events = [e for sublist in all_events for e in sublist]
        events_by_date = {}
        
        for event in flat_events:
            if event.date not in events_by_date:
                events_by_date[event.date] = []
            events_by_date[event.date].append(event)
            
        merged_results = []
        for date, events in events_by_date.items():
            if len(events) > 1:
                events_text = "\n".join([
                    f"- {e.data_type} ({e.category}): {e.value} (Severity: {e.severity})\n  Impact: {e.potential_market_impact}"
                    for e in events
                ])
                
                synthesis = self.llm.invoke(
                    f"""Multiple economic anomalies occurred on {date}:
                    {events_text}
                    
                    Please provide:
                    1. A synthesis of how these events are related
                    2. Their combined economic significance
                    3. The potential compound effect on financial markets
                    4. An overall severity rating (0.0-1.0) for this date's events"""
                )
                
                merged_results.append({
                    "date": date,
                    "events": [{
                        "type": e.data_type,
                        "value": e.value,
                        "category": e.category,
                        "severity": e.severity,
                        "market_impact": e.potential_market_impact
                    } for e in events],
                    "summary": synthesis.content,
                    "overall_severity": max(e.severity for e in events)
                })
            else:
                event = events[0]
                merged_results.append({
                    "date": date,
                    "events": [{
                        "type": event.data_type,
                        "value": event.value,
                        "category": event.category,
                        "severity": event.severity,
                        "market_impact": event.potential_market_impact
                    }],
                    "summary": event.description,
                    "overall_severity": event.severity
                })
                
        return sorted(merged_results, key=lambda x: x["date"])

    def save_output(self, output: List[str], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output))
        except Exception as e:
            logging.error(f"Failed to save output to {filename}: {e}", exc_info=True)
            raise

def analyze_economic_data(data_dir: str = "Economic_EVENTS") -> List[Dict[str, Any]]:
    """Main function to orchestrate parallel analysis and merging"""
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.json')]
    
    analyzers = [FileAnalyzer(f) for f in json_files]
    
    with ThreadPoolExecutor() as executor:
        all_results = list(executor.map(lambda a: a.analyze(), analyzers))
        
    merger = ResultMerger()
    final_results = merger.merge_results(all_results)
    
    return final_results

if __name__ == "__main__":
    try:
        logging.info("Starting economic data analysis")
        weird_events = analyze_economic_data()
        
        if not weird_events:
            logging.warning("No events were found in the analysis")
            exit(1)
            
        # Create a more detailed and structured output
        output = []
        output.append("Economic Events Analysis Report")
        output.append("=" * 50 + "\n")
        
        # Add a summary of high-severity events
        high_severity_events = [e for e in weird_events if e["overall_severity"] >= 0.7]
        if high_severity_events:
            logging.info(f"Found {len(high_severity_events)} high-severity events")
            output.append("HIGH PRIORITY EVENTS")
            output.append("-" * 20)
            for event in high_severity_events:
                output.append(f"Date: {event['date']} (Severity: {event['overall_severity']:.2f})")
                output.append("Events:")
                for e in event['events']:
                    output.append(f"  - {e['type']} ({e['category']}): {e['value']}")
                    output.append(f"    Market Impact: {e['market_impact']}")
                output.append(f"Summary: {event['summary']}\n")
            output.append("=" * 50 + "\n")
        
        # Main report with all events
        output.append("FULL EVENT LOG")
        output.append("-" * 20)
        for event in weird_events:
            output.append(f"Date: {event['date']} (Severity: {event['overall_severity']:.2f})")
            output.append("Economic Indicators:")
            for e in event['events']:
                output.append(f"  - Category: {e['category']}")
                output.append(f"    Indicator: {e['type']}")
                output.append(f"    Value: {e['value']}")
                output.append(f"    Severity: {e['severity']:.2f}")
                output.append(f"    Market Impact: {e['market_impact']}")
            output.append(f"Analysis: {event['summary']}")
            output.append("-" * 30 + "\n")
        
        # Analyze for major market-moving events
        merger = ResultMerger()
        major_market_events = merger.analyze_market_impact(weird_events)
        
        # Create major market events report
        market_output = []
        market_output.append("Major Market-Moving Events Report")
        market_output.append("=" * 50 + "\n")
        
        for event in major_market_events:
            market_output.append(f"Date: {event['date']}")
            market_output.append(f"Market Direction: {event['market_direction']}")
            market_output.append(f"Confidence: {event['confidence']:.2f}")
            market_output.append(f"Explanation: {event['explanation']}")
            market_output.append("\nContributing Factors:")
            for e in event['events']:
                market_output.append(f"  - {e['type']}: {e['value']}")
            market_output.append(f"\nDetailed Analysis: {event['original_summary']}")
            market_output.append("-" * 30 + "\n")
        
        # Create simple date list
        date_output = []
        date_output.append("Significant Market-Moving Dates")
        date_output.append("=" * 30 + "\n")
        for event in major_market_events:
            date_output.append(f"{event['date']} | {event['market_direction']} | {event['confidence']:.2f}")
        
        # Save all outputs
        try:
            merger.save_output(output, 'economic_analysis.txt')
            
            with open('economic_events.json', 'w') as f:
                json.dump(weird_events, f, indent=2)
                
            merger.save_output(market_output, 'major_market_events.txt')
                
            merger.save_output(date_output, 'market_moving_dates.txt')
                
            with open('major_market_events.json', 'w') as f:
                json.dump(major_market_events, f, indent=2)
                
            logging.info("Analysis complete! Results saved successfully")
            print("Analysis complete! Results saved to:")
            print("1. economic_analysis.txt (Human-readable report)")
            print("2. economic_events.json (Structured data for further analysis)")
            print("3. major_market_events.txt (Detailed market impact analysis)")
            print("4. market_moving_dates.txt (Simple list of significant dates)")
            print("5. major_market_events.json (Structured market event data)")
            print("6. economic_analysis.log (Processing log with warnings and errors)")
            
        except Exception as e:
            logging.error("Failed to save output files", exc_info=True)
            print("Error: Failed to save output files. Check the log for details.")
            
    except Exception as e:
        logging.error("Analysis failed", exc_info=True)
        print("Error: Analysis failed. Check the log for details.") 