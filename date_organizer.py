from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI

@dataclass
class WeirdEvent:
    date: str
    description: str
    data_type: str
    value: float
    source_file: str

class FileAnalyzer:
    def __init__(self, filename: str):
        self.filename = filename
        # Remove temperature setting since o1-mini only supports default
        self.llm = ChatOpenAI(model="o1-mini")
    
    def analyze(self) -> List[WeirdEvent]:
        with open(self.filename, 'r') as f:
            data = json.load(f)
            
        # Format the data points for LLM analysis
        data_points = "\n".join([
            f"Date: {point['date']}, Value: {point['value']}"
            for point in data["data"]
        ])
        
        # Have LLM identify weird events with simpler format
        analysis = self.llm.invoke(
            f"""Analyze this time series data for {data['name']} ({data['unit']}) and identify unusual points.
            
            {data_points}

            For each unusual point, respond in EXACTLY this format (one per line):
            POINT: [DATE] | [VALUE] | [DESCRIPTION]
            
            Example format:
            POINT: 2024-01-01 | 100.5 | Unexpected 50% increase
            """
        )
        
        # Parse LLM response into WeirdEvent objects with more robust parsing
        weird_events = []
        for line in analysis.content.split('\n'):
            if line.strip().startswith('POINT:'):
                try:
                    # Remove 'POINT:' and split by |
                    parts = line.replace('POINT:', '').strip().split('|')
                    if len(parts) == 3:
                        date = parts[0].strip()
                        value = float(parts[1].strip())
                        description = parts[2].strip()
                        
                        weird_events.append(WeirdEvent(
                            date=date,
                            description=description,
                            data_type=data["name"],
                            value=value,
                            source_file=self.filename
                        ))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    continue
                
        return weird_events

class ResultMerger:
    def __init__(self):
        # Use o1-mini model ID directly with ChatOpenAI
        self.llm = ChatOpenAI(model="o1-mini")
        
    def merge_results(self, all_events: List[List[WeirdEvent]]) -> List[Dict[str, Any]]:
        # Flatten events
        flat_events = [e for sublist in all_events for e in sublist]
        
        # Group by date
        events_by_date = {}
        for event in flat_events:
            if event.date not in events_by_date:
                events_by_date[event.date] = []
            events_by_date[event.date].append(event)
            
        merged_results = []
        for date, events in events_by_date.items():
            # Have LLM synthesize multiple events on same date
            if len(events) > 1:
                events_text = "\n".join([
                    f"- {e.data_type}: {e.value} ({e.description})"
                    for e in events
                ])
                
                synthesis = self.llm.invoke(
                    f"""Multiple economic anomalies occurred on {date}:
                    {events_text}
                    
                    Please provide a brief synthesis of how these events might be related and their potential combined economic significance."""
                )
                
                merged_results.append({
                    "date": date,
                    "events": [{"type": e.data_type, "value": e.value} for e in events],
                    "summary": synthesis.content
                })
            else:
                event = events[0]
                merged_results.append({
                    "date": date,
                    "events": [{"type": event.data_type, "value": event.value}],
                    "summary": event.description
                })
                
        return sorted(merged_results, key=lambda x: x["date"])

def analyze_economic_data(data_dir: str = "Economic_EVENTS") -> List[Dict[str, Any]]:
    """Main function to orchestrate parallel analysis and merging"""
    
    # Get all JSON files
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.json')]
    
    # Create analyzers for each file
    analyzers = [FileAnalyzer(f) for f in json_files]
    
    # Run analyses using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        all_results = list(executor.map(lambda a: a.analyze(), analyzers))
        
    # Merge results
    merger = ResultMerger()
    final_results = merger.merge_results(all_results)
    
    return final_results

# Example usage:
if __name__ == "__main__":
    weird_events = analyze_economic_data()
    
    # Format the output in a readable way
    output = []
    output.append("Economic Events Analysis Report")
    output.append("=" * 30 + "\n")
    
    for event in weird_events:
        output.append(f"Date: {event['date']}")
        output.append("Events:")
        for e in event['events']:
            output.append(f"  - {e['type']}: {e['value']}")
        output.append(f"Summary: {event['summary']}")
        output.append("-" * 30 + "\n")
    
    # Save to file
    with open('economic_analysis.txt', 'w') as f:
        f.write('\n'.join(output))
    
    print("Analysis complete! Results saved to 'economic_analysis.txt'")
