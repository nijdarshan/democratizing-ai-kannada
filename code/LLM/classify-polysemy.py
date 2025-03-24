import pandas as pd
import json
import time
import logging
import random
from pathlib import Path
from string import Template
from typing import List, Dict, Optional
from datetime import datetime
from anthropic import AnthropicVertex
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.oauth2 import service_account
from google.cloud import aiplatform

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('polysemy_classifier.log', encoding='utf-8'),  # Added encoding
        logging.StreamHandler()
    ]
)

# Configure console output for Unicode
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class JSONError(Exception):
    """Custom exception for JSON-related errors"""
    pass

class PolysemyClassifier:
    def __init__(
        self,
        project_id: str,
        service_account_path: str,
        location: str = "us-east5",
        max_sentences: int = 10,
        output_dir: str = "classifications",
        model: str = "claude-3-5-sonnet-v2@20241022"
    ):
        """Initialize the classifier with configuration parameters"""
        # Load service account credentials
        try:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Vertex AI
            aiplatform.init(
                project=project_id,
                location=location,
                credentials=credentials
            )
            
            self.client = AnthropicVertex(
                project_id=project_id,
                region=location,
                credentials=credentials
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize credentials: {str(e)}")
            raise
            
        self.max_sentences = max_sentences
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load template
        try:
            with open("prompt-template.txt", "r", encoding='utf-8') as f:
                self.prompt_template = Template(f.read())
        except FileNotFoundError:
            logger.error("Prompt template file not found")
            raise

    def prepare_sentences(self, word_sentences: pd.DataFrame) -> List[Dict]:
        """Prepare and sample sentences for classification"""
        sentences = word_sentences['sentence'].tolist()
        if len(sentences) > self.max_sentences:
            indices = random.sample(range(len(sentences)), self.max_sentences)
            sentences = [sentences[i] for i in sorted(indices)]
        else:
            indices = list(range(len(sentences)))
            
        return [{"index": idx, "text": sent} for idx, sent in zip(indices, sentences)]

    def validate_json_response(self, data: Dict) -> bool:
        """Validate the structure of the JSON response"""
        required_fields = ['word', 'meanings', 'classifications']
        if not all(field in data for field in required_fields):
            return False
            
        if not isinstance(data['meanings'], dict):
            return False
            
        if not isinstance(data['classifications'], list):
            return False
            
        for classification in data['classifications']:
            required_class_fields = ['index', 'sentence', 'selected_meaning', 'contextual_analysis']
            if not all(field in classification for field in required_class_fields):
                return False
                
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIError, JSONError))
    )
    def classify_word(self, word: str, prepared_sentences: List[Dict]) -> Dict:
        """Classify a word's meanings in the given sentences"""
        try:
            # Format sentences for prompt
            formatted_sentences = "\n".join(
                f"{i+1}. {s['text']}" for i, s in enumerate(prepared_sentences)
            )
            
            # Prepare prompt
            prompt = self.prompt_template.substitute(
                word=word,
                sentences=formatted_sentences
            )
            
            # Make API call
            response = self.client.messages.create(
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
            )
            
            # Extract and parse JSON
            content = response.content[0].text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise JSONError("No JSON found in response")
                
            json_content = content[json_start:json_end]
            result = json.loads(json_content)
            
            # Validate response
            if not self.validate_json_response(result):
                raise JSONError("Invalid JSON structure in response")
                
            return result
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise APIError(f"Rate limit exceeded: {str(e)}")
            elif "json" in str(e).lower():
                raise JSONError(f"JSON error: {str(e)}")
            else:
                raise APIError(f"API error: {str(e)}")

    def process_file(self, input_file: str):
        """Process the input CSV file containing words and sentences"""
        try:
            df = pd.read_csv(input_file)
            
            # Create timestamp for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.output_dir / f"run_{timestamp}"
            run_dir.mkdir(exist_ok=True)
            
            # Process each word
            for word, word_sentences in df.groupby('word'):
                output_file = run_dir / f"{word}_classification.json"
                
                if output_file.exists():
                    logger.info(f"Skipping {word} - already processed")
                    continue
                
                logger.info(f"Processing word: {word}")
                
                try:
                    # Prepare and classify
                    prepared_sentences = self.prepare_sentences(word_sentences)
                    result = self.classify_word(word, prepared_sentences)
                    
                    # Save results
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Successfully processed {word}")
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Failed to process word {word}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to process file: {str(e)}")
            raise

def main():
    # Configuration
    config = {
        "project_id": "utopian-saga-438818-j0",
        "service_account_path": "D:/Dwds/utopian-saga-438818-j0-d45170666985.json",
        "input_file": "subset.csv",
        "max_sentences": 10,
        "output_dir": "classifications"
    }
    
    try:
        # Make sure you have the required dependencies
        try:
            from google.cloud import aiplatform
        except ImportError:
            logger.error("Please install google-cloud-aiplatform: pip install google-cloud-aiplatform")
            raise
            
        classifier = PolysemyClassifier(
            project_id=config["project_id"],
            service_account_path=config["service_account_path"],
            max_sentences=config["max_sentences"],
            output_dir=config["output_dir"]
        )
        
        classifier.process_file(config["input_file"])
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()