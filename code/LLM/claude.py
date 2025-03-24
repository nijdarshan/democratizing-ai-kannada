import anthropic
import pandas as pd
import json
import logging
from datetime import datetime
import os
import time

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key="")

# Load the CSV file and organize input by words and sentences
df = pd.read_csv("subsetmax.csv")
grouped_sentences = df.groupby('word')['sentence'].apply(list).to_dict()

# Define system and user messages for API
system_message = """
As a linguist in Kannada, analyze the meanings of a polysemous word across multiple sentences and selecting the most contextually appropriate meaning from only two possible senses. Output must strictly follow the JSON format.
"""

# Function to prepare the prompt with examples and a specific word's sentences
def create_message(word, sentences):
    examples = """
<examples>
  <example>
    <word>
    pitch
    </word>
    <sentences>
    1. "The singer hit a perfect pitch during her performance."
    2. "The company delivered a compelling pitch to attract investors."
    3. "They marked the lines on the pitch for the soccer game."
    </sentences>
    <ideal_output>
    {
      "word": "pitch",
      "classifications": [
        {
          "index": 0,
          "sentence": "The singer hit a perfect pitch during her performance.",
          "selected_meaning": "1"
        },
        {
          "index": 1,
          "sentence": "The company delivered a compelling pitch to attract investors.",
          "selected_meaning": "2"
        },
        {
          "index": 2,
          "sentence": "They marked the lines on the pitch for the soccer game.",
          "selected_meaning": "3"
        }
      ]
    }
    </ideal_output>
  </example>
</examples>
    """

    # Create the sentences list first
    sentence_list = '\n'.join(f'{i+1}. "{sent}"' for i, sent in enumerate(sentences[:10]))

    # Prepare specific prompt for the word
    analysis = f"""
Input Word: {word}

Sentences to analyze:
{sentence_list}

Instructions:
1. From the 2 possible meanings of the word "{word}", assigning a unique number to each meaning.
2. If all sentences belong to one sense, classify accordingly.
3. For each sentence, identify the meaning number that best fits the context.
4. Provide output in strict JSON format with no additional text.

Expected JSON structure:
{{
  "word": "{word}",
  "classifications": [
    {{
      "index": 0,
      "sentence": "Sentence text",
      "selected_meaning": "1"
    }}
  ]
}}
"""

    return {
        "system": system_message,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": examples}]},
            {"role": "user", "content": [{"type": "text", "text": analysis}]}
        ]
    }

def save_response(word, batch_num, response_data):
    """Save individual response to JSON file"""
    try:
        filename = os.path.join(output_dir, f"{word}_batch_{batch_num}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved response for {word} batch {batch_num} to {filename}")
    except Exception as e:
        logging.error(f"Error saving response for {word} batch {batch_num}: {str(e)}")

def process_word_batch(word, sentences, batch_num):
    """Process a batch of sentences for a word"""
    try:
        message_data = create_message(word, sentences)
        
        # API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=5000,
                    temperature=0,
                    **message_data
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed for {word} batch {batch_num}: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        response_data = {
            "word": word,
            "batch": batch_num,
            "sentences": sentences,
            "output": response.content[0].text
        }
        
        save_response(word, batch_num, response_data)
        return response_data
        
    except Exception as e:
        logging.error(f"Error processing {word} batch {batch_num}: {str(e)}")
        return None

# Process each word and gather results
results = []
batch_size = 10

for word, sentences in grouped_sentences.items():
    logging.info(f"Processing word: {word} with {len(sentences)} sentences")
    
    # Process sentences in batches of 10
    for i in range(0, len(sentences), batch_size):
        batch_num = i // batch_size + 1
        batch_sentences = sentences[i:i + batch_size]
        
        logging.info(f"Processing batch {batch_num} for word {word}")
        result = process_word_batch(word, batch_sentences, batch_num)
        
        if result:
            results.append(result)
        
        # Add a small delay between batches to avoid rate limiting
        time.sleep(1)

# Save overall results to JSON file
try:
    overall_results_file = os.path.join(output_dir, 'all_results.json')
    with open(overall_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved all results to {overall_results_file}")
except Exception as e:
    logging.error(f"Error saving overall results: {str(e)}")

logging.info("Processing completed")