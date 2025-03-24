from anthropic import AnthropicVertex
from google.oauth2 import service_account
import time
import random

# Constants
PROJECT_ID = "utopian-saga-438818-j0"
LOCATION = "us-east5"
MODEL = "claude-3-5-sonnet-v2@20241022"
CREDENTIALS_FILE = "D:/Dwds/utopian-saga-438818-j0-d45170666985.json"

def test_prompt():
    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    # Initialize client with credentials
    client = AnthropicVertex(
        project_id=PROJECT_ID,
        region=LOCATION,
        credentials=credentials
    )

    max_retries = 3
    initial_delay = 60  # 60 seconds initial delay
    
    # Test prompt
    full_prompt = "Say hello in one sentence."
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            
            # Using exact same message format as working code
            message = client.messages.create(
                max_tokens=1024,
                messages=[{
                    "role": "user", 
                    "content": full_prompt
                }],
                model=MODEL
            )
            
            print("Success! Response:")
            print(message.content[0].text)
            return
            
        except Exception as e:
            print(f"Error occurred: {e}")
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 10)
            print(f"Waiting {delay:.2f} seconds before retry...")
            time.sleep(delay)
    
    print("Failed after all retries")

if __name__ == "__main__":
    test_prompt()