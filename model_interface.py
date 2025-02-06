from typing import Dict, List, Optional, Union
import google.generativeai as genai
from openai import OpenAI
import warnings
import urllib3
from config import (
    GOOGLE_API_KEY, 
    OPENAI_API_KEY, 
    DEEPSEEK_API_KEY,
    validate_credentials
)
from prompt_creator2 import PromptCreator
import argparse
import os
import datetime

warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

class ModelInterface:
    def __init__(self):
        self.initialize_clients()
        self.prompt_creator = PromptCreator()
        # Create base output directory if it doesn't exist
        self.base_output_dir = "model_outputs"
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.current_ticker = None
        self.output_dir = None
        
    def initialize_clients(self):
        """Initialize API clients for each model provider"""
        # Initialize Google/Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536
        }
        
        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Deepseek
        self.deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    def call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.0-flash-thinking-exp-01-21"
    ) -> str:
        """Call Gemini models"""
        print(f"\nDebug: Calling Gemini model: {model}")
        try:
            # Initialize model with specific config
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=self.gemini_config
            )
            
            if system_prompt:
                print("Debug: Using chat mode with system prompt")
                chat = model_instance.start_chat(history=[])
                response = chat.send_message(prompt)
            else:
                print("Debug: Using generate_content mode")
                response = model_instance.generate_content(prompt)
                
            print("Debug: Successfully got response from Gemini")
            return response.text
            
        except Exception as e:
            print(f"Debug: Error calling Gemini model {model}: {e}")
            return None

    def call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "o1-mini"
    ) -> str:
        """Call OpenAI models"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=8192
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None

    def call_deepseek(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "deepseek-reasoner"
    ) -> str:
        """Call Deepseek models"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling Deepseek: {e}")
            return None

    def save_interaction(self, provider: str, prompt: str, system_prompt: Optional[str], response: str, model: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """Save the model interaction to files with descriptive names"""
        print(f"\nDebug: Attempting to save interaction...")
        print(f"Debug: Output directory: {self.output_dir}")
        print(f"Debug: Provider: {provider}")
        print(f"Debug: Model: {model}")
        print(f"Debug: Response length: {len(response) if response else 'None'}")
        
        try:
            # Use provided indicator and dates directly from arguments
            indicator = getattr(self, 'passed_indicator', None) or "unknown"
            start_date = start_date or "unknown"
            end_date = end_date or "unknown"

            model_name = model if model else provider
            mode = "thinking" if "thinking" in model_name.lower() else "base"

            # If start_date and end_date are the same, include the month once
            if start_date == end_date:
                base_filename = f"{provider}_{mode}_{indicator}_{start_date}"
            else:
                base_filename = f"{provider}_{mode}_{indicator}_{start_date}_{end_date}"
            
            # Save prompt
            prompt_filename = os.path.join(self.output_dir, f"{base_filename}_prompt.txt")
            print(f"Debug: Writing prompt to {prompt_filename}")
            with open(prompt_filename, "w", encoding="utf-8") as f:
                if system_prompt:
                    f.write("=== System Prompt ===\n")
                    f.write(system_prompt)
                    f.write("\n\n=== User Prompt ===\n")
                f.write(prompt)
                
            # Save response
            response_filename = os.path.join(self.output_dir, f"{base_filename}_response.txt")
            print(f"Debug: Writing response to {response_filename}")
            with open(response_filename, "w", encoding="utf-8") as f:
                f.write(response)
                
            print("Debug: Successfully saved interaction files")
            
        except Exception as e:
            print(f"Debug: Error saving interaction: {e}")
            import traceback
            print("Debug: Full traceback:")
            print(traceback.format_exc())
            raise

    def generate_response(
        self,
        provider: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Unified interface to generate responses from any model
        
        Args:
            provider: One of 'gemini', 'openai', or 'deepseek'
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Optional specific model name
        """
        print(f"\nDebug: Generating response for provider: {provider}")
        provider_map = {
            'gemini': self.call_gemini,
            'openai': self.call_openai,
            'deepseek': self.call_deepseek
        }
        
        if provider not in provider_map:
            raise ValueError(f"Unknown provider: {provider}. Must be one of {list(provider_map.keys())}")
            
        response = provider_map[provider](
            prompt=prompt,
            system_prompt=system_prompt,
            model=model if model else None
        )
        
        print(f"Debug: Got response: {response is not None}")
        print(f"Debug: Response type: {type(response)}")
        print(f"Debug: Response length: {len(response) if response else 'None'}")
        
        # Save the interaction
        if response is not None:
            try:
                self.save_interaction(provider, prompt, system_prompt, response, model)
            except Exception as e:
                print(f"Debug: Failed to save interaction: {e}")
                import traceback
                print("Debug: Full traceback:")
                print(traceback.format_exc())
            
        return response

    def set_ticker_output_dir(self, ticker: str):
        """Set up the output directory for a specific ticker"""
        self.current_ticker = ticker
        self.output_dir = os.path.join(self.base_output_dir, f"{ticker}_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_response_with_created_prompt(
        self,
        provider: str,
        ticker: str,
        indicator: str,
        include_econ_events: bool = False,
        model: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """
        Generate response using a prompt created by PromptCreator
        """
        try:
            # Set up ticker-specific output directory
            self.set_ticker_output_dir(ticker)
            
            # Get the filled prompt from PromptCreator, using provided start and end dates
            filled_prompt = self.prompt_creator.create_prompt(
                model=model or provider,
                ticker=ticker,
                indicator=indicator,
                include_econ_events=include_econ_events,
                start_date=start_date,
                end_date=end_date
            )
            
            # Generate response based on provider
            if provider.startswith('gemini'):
                response = self.call_gemini(
                    prompt=filled_prompt,
                    model=model or "gemini-2.0-flash-thinking-exp-01-21"
                )
                if response is not None:
                    # Pass the indicator from argument via temporary attribute
                    self.passed_indicator = indicator
                    self.save_interaction(provider, filled_prompt, None, response, model, start_date, end_date)
                    del self.passed_indicator
                return response
            else:
                user_prompt, system_prompt = filled_prompt
                response = self.generate_response(
                    provider=provider,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model
                )
                if response is not None:
                    # Pass the indicator from argument via temporary attribute
                    self.passed_indicator = indicator
                    self.save_interaction(provider, user_prompt, system_prompt, response, model, start_date, end_date)
                    del self.passed_indicator
                return response
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Generate model responses using formatted prompts')
    parser.add_argument('--provider', type=str, required=True,
                       choices=['gemini', 'openai', 'deepseek'],
                       help='Model provider to use')
    parser.add_argument('--model', type=str,
                       choices=["claude-sonnet-3.5", "o1-mini", "gemini-exp-1206", "gemini-2.0-flash-thinking-exp-01-21"],
                       help='Specific model to use (optional)')
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--indicator', type=str, required=True,
                       help='Technical indicator to analyze')
    parser.add_argument('--include_market_events', type=str2bool, default=False,
                       help='Whether to include market events data')
    parser.add_argument('--supercharge', action='store_true',
                       help='Enable supercharge mode')

    args = parser.parse_args()
    
    # Initialize model interface
    interface = ModelInterface()
    
    try:
        if args.supercharge:
            # Supercharge mode: iterate through all month folders in the ticker's historical directory
            historical_dir = f"{args.ticker}_HISTORICAL"
            if not os.path.isdir(historical_dir):
                print(f"Historical data folder '{historical_dir}' not found.")
                return 1

            # Get list of month directories matching YYYY_MM
            month_dirs = [d for d in os.listdir(historical_dir) if os.path.isdir(os.path.join(historical_dir, d)) and len(d)==7 and d[4]=='_']
            if not month_dirs:
                print(f"No month directories found in '{historical_dir}'.")
                return 1

            # Sort month directories
            month_dirs.sort()

            combined_reports = []
            for month in month_dirs:
                print(f"Processing month: {month}")
                # Call generate_response_with_created_prompt with start_date and end_date set to the same month
                response = interface.generate_response_with_created_prompt(
                    provider=args.provider,
                    ticker=args.ticker,
                    indicator=args.indicator,
                    include_econ_events=args.include_market_events,
                    model=args.model,
                    start_date=month,
                    end_date=month
                )
                if response is None:
                    print(f"No response for month {month}")
                else:
                    header = f"\n{'='*80}\nReport for {month}\n{'='*80}\n"
                    combined_reports.append(header + response + "\n")

            # First, create a concatenated text from the individual monthly reports
            combined_text = "\n".join(combined_reports)

            # Create a prompt for the LLM to synthesize a final comprehensive report
            final_prompt = (
                f"You are a comprehensive analyst. Analyze the monthly reports for ticker {args.ticker} with indicator {args.indicator}. "
                "Below are the reports generated for each month. Provide a final comprehensive report that includes all critical details, patterns, trends, insights, and suggestions. Do not miss any important information.\n\n"
                f"Monthly Reports:\n{combined_text}\n\nFinal Report:"
            )

            # Call the Gemini model to generate the final comprehensive report
            final_response = interface.call_gemini(
                prompt=final_prompt,
                system_prompt=None,
                model="gemini-2.0-flash-thinking-exp-01-21"
            )

            # Create final combined report filename
            provider = args.provider
            model_used = args.model if args.model else args.provider
            mode = "thinking" if "thinking" in model_used.lower() else "base"
            first_month = month_dirs[0]
            last_month = month_dirs[-1]
            final_filename = os.path.join(interface.output_dir, f"{provider}_{mode}_{args.indicator}_{first_month}_{last_month}_final_combined_response.txt")

            with open(final_filename, "w", encoding="utf-8") as f:
                f.write(final_response)

            print(f"\nFinal comprehensive report saved to {final_filename}")
            return 0
        else:
            # Normal mode: ask interactively for start and end month
            start_date = input("Enter start month (YYYY_MM): ")
            end_date = input("Enter end month (YYYY_MM): ")

            response = interface.generate_response_with_created_prompt(
                provider=args.provider,
                ticker=args.ticker,
                indicator=args.indicator,
                include_econ_events=args.include_market_events,
                model=args.model,
                start_date=start_date,
                end_date=end_date
            )

            print("\nModel Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)

            print("\nInteraction saved to files in the model_outputs directory")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    '''
    Example usage:
    python model_interface.py --provider gemini --model gemini-2.0-flash-thinking-exp-01-21 --ticker NVDA --indicator bbands --include_market_events False
    python model_interface.py --provider openai --ticker NVDA --indicator bop --include_market_events False
    python model_interface.py --provider deepseek --ticker NVDA --indicator bop --include_market_events False

    For supercharge mode:
    python model_interface.py --provider gemini --model gemini-2.0-flash-thinking-exp-01-21 --ticker NVDA --indicator bbands --include_market_events False --supercharge 
    '''
    import sys
    sys.exit(main()) 