import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google AI credentials
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')

# OpenAI credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Alpha Vantage credentials (for stock data)
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

# Polygon.io credentials (if you're using it)
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

# Deepseek credentials
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# You can group related credentials into dictionaries if needed
google_credentials = {
    'api_key': GOOGLE_API_KEY,
    'project_id': GOOGLE_PROJECT_ID
}

# Optional: Function to validate that required keys are present
def validate_credentials():
    required_keys = ['GOOGLE_API_KEY', 'GOOGLE_PROJECT_ID', 'DEEPSEEK_API_KEY']  # Add keys as needed
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}") 