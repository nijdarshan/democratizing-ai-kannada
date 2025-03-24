import anthropic
import pandas as pd
import json
import logging
import os
import time
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

def parse_existing_classifications(results_file):
    """Parse existing classification results file and return processed sentences"""
    processed_data = {"processed_sentences": {}}
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by the separator
            classifications = content.split("\n" + "="*50 + "\n")
            
            for classification in classifications:
                if not classification.strip():
                    continue
                    
                try:
                    data = json.loads(classification.strip())
                    word = data["word"]
                    
                    if word not in processed_data["processed_sentences"]:
                        processed_data["processed_sentences"][word] = []
                        
                    # Add all sentences from this batch
                    for item in data["classifications"]:
                        if item["sentence"] not in processed_data["processed_sentences"][word]:
                            processed_data["processed_sentences"][word].append(item["sentence"])
                            
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse classification entry: {classification[:100]}...")
                    continue
                    
        logging.info(f"Found existing classifications for {len(processed_data['processed_sentences'])} words")
        for word, sentences in processed_data['processed_sentences'].items():
            logging.info(f"Word '{word}': {len(sentences)} sentences processed")
            
    except FileNotFoundError:
        logging.info("No existing classification results file found")
        
    return processed_data

class SentenceClassifier:
    def __init__(self, api_keys):
        # Initialize with multiple API keys
        self.api_keys = api_keys
        self.current_key_index = 0
        self.clients = [anthropic.Anthropic(api_key=key) for key in api_keys]
        
        self.state_file = "classification_state.json"
        self.processed_data = parse_existing_classifications("classification_results.txt")
        self.merge_state_data()
        self.retry_count = 0
        self.max_retries = 5
        self.base_wait = 5  # Base wait time in seconds
        
        # Cache prompt components
        self.system_message = "You are an expert computational linguist specializing in polysemy and word sense disambiguation for Kannada language."
        self.examples = '''
        Example 1:
        Word: ಕಾಯಿ
        Meanings:
        Sense 1:
        Kannada: ಮರದ ಫಲ
        English: fruit
        Sense 2:
        Kannada: ಆಟದ ಪಾತ್ರೆ
        English: game piece
        
        Sentences:
        1. ಮರದಿಂದ ಕಾಯಿ ಬಿದ್ದಿತು
        2. ಚದುರಂಗದಲ್ಲಿ ಕಾಯಿ ಚಲಿಸಿದ
        
        Output: {"labels": [1, 2]}

        Example 2:
        Word: bank
        Meanings:
        Sense 1:
        Kannada: ಹಣಕಾಸು ಸಂಸ್ಥೆ
        English: financial institution
        Sense 2:
        Kannada: ನದಿ ದಡ
        English: river bank
        
        Sentences:
        1. I withdrew money from the bank
        2. We sat by the river bank
        
        Output: {"labels": [1, 2]}
        '''
        
        self.instruction_template = '''
        Classify the given Kannada sentences based on their polysemous word senses.

        Target Word: {word}
        
        Meanings:
        {meanings}

        Classification Guidelines:
        1. Semantic Analysis:
           - Consider the complete semantic context
           - Analyze collocations and word associations
           - Look for domain-specific indicators
           - Consider both Kannada and English meanings
        
        2. Special Cases:
           - Assign -1 if:
             * Word is used as a proper noun (person/place/movie name)
             * Sentence has significant typos
             * Sentence is grammatically invalid
           
           - Assign 0 if:
             * Cannot determine sense confidently
             * Context is insufficient
             * Sentence is ambiguous
        
        3. Linguistic Considerations:
           - Consider Kannada's SOV structure
           - Account for morphological variations
           - Handle idiomatic expressions
           - Consider register variations

        Sentences to classify:
        {sentences}

        Return ONLY a JSON with numeric labels. Example: {{"labels": [1, 2]}}
        - Labels must be numbers only (-1, 0, 1, 2, 3, or 4)
        - Return exactly one label per sentence
        - Return ONLY the JSON, no other text
        '''
        
        self.current_word = None
        self.current_meanings_formatted = None

    def get_next_client(self):
        """Rotate to next API client"""
        client = self.clients[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.clients)
        return client
            
    def merge_state_data(self):
        """Merge any additional state data with existing classifications"""
        if Path(self.state_file).exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                
            # Merge state data with existing classifications
            for word, sentences in state_data["processed_sentences"].items():
                if word not in self.processed_data["processed_sentences"]:
                    self.processed_data["processed_sentences"][word] = []
                for sentence in sentences:
                    if sentence not in self.processed_data["processed_sentences"][word]:
                        self.processed_data["processed_sentences"][word].append(sentence)

    def load_state(self):
        """Load previously processed sentences from state file"""
        if Path(self.state_file).exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"processed_sentences": {}}
    
    def save_state(self, word, sentences):
        """Save processed sentences to state file"""
        if word not in self.processed_data["processed_sentences"]:
            self.processed_data["processed_sentences"][word] = []
        self.processed_data["processed_sentences"][word].extend(sentences)
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)

    def format_meanings(self, meanings):
        """Format meanings once per word"""
        formatted = ""
        for idx, meaning in meanings.items():
            formatted += f"\nSense {idx}:\n"
            formatted += f"Kannada: \"{meaning['kannada']}\"\n"
            formatted += f"English: \"{meaning['english']}\"\n"
        return formatted

    def create_message(self, word, meanings, sentences):
        # Only format meanings if this is a new word
        if self.current_word != word:
            self.current_word = word
            self.current_meanings_formatted = self.format_meanings(meanings)
        
        # Format just the sentences for this batch
        formatted_sentences = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
        
        # Fill in the template with cached meanings
        instructions = self.instruction_template.format(
            word=word,
            meanings=self.current_meanings_formatted,
            sentences=formatted_sentences
        )
        
        return {
            "system": self.system_message,
            "messages": [
                {"role": "user", "content": self.examples},
                {"role": "user", "content": instructions}
            ]
        }

    @retry(
        wait=wait_exponential(multiplier=5, min=5, max=300),  # Reduced wait times
        stop=stop_after_attempt(5)
    )
    def classify_batch(self, word, meanings, sentences, batch_index):
        try:
            message_data = self.create_message(word, meanings, sentences)

            # Get next client in rotation
            client = self.get_next_client()
            
            # Add delay based on recent failures
            sleep_time = self.base_wait * (1.5 ** self.retry_count)
            logging.info(f"Using API key {self.current_key_index + 1}, waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                **message_data
            )
            
            # Reset retry count on success
            self.retry_count = 0
            
            # Extract just the JSON part from response
            response_text = response.content[0].text.strip()
            
            # Log token usage if available
            if hasattr(response, 'usage'):
                logging.info(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
            
            # Handle potential formatting issues
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON if surrounded by other text
                import re
                json_match = re.search(r'\{.*\}', response_text)
                if json_match:
                    response_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from response")
            
            self.save_results(word, batch_index, response_data, sentences)
            return response_data.get("labels", [])
            
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                self.retry_count += 1
                logging.warning(f"Rate limit hit. Retry attempt {self.retry_count}")
                raise  # Let the retry decorator handle it
            logging.error(f"Error in classification batch {batch_index} for word {word}: {str(e)}")
            logging.error(f"Response text: {response_text if 'response_text' in locals() else 'No response'}")
            return []

    def save_results(self, word, batch_index, results, sentences):
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output = {
                "timestamp": timestamp,
                "word": word,
                "batch": batch_index,
                "classifications": [
                    {
                        "sentence": sent,
                        "label": label
                    }
                    for sent, label in zip(sentences, results["labels"])
                ]
            }
            
            with open("classification_results.txt", "a", encoding="utf-8") as f:
                f.write("\n" + "="*50 + "\n")
                f.write(json.dumps(output, ensure_ascii=False, indent=2))
                f.write("\n")
                
            # Add state saving after saving results
            self.save_state(word, sentences)
            
        except Exception as e:
            logging.error(f"Error saving results for word {word}, batch {batch_index}: {str(e)}")

def create_meanings_dict(row):
    meanings = {}
    for i in range(1, 5):  # Check all 4 possible meanings
        kannada_key = f'kannada_meaning_{i}'
        english_key = f'english_meaning_{i}'

        # Only add if both Kannada and English meanings exist and are not empty
        if pd.notna(row.get(kannada_key)) and pd.notna(row.get(english_key)) and \
           str(row[kannada_key]).strip() and str(row[english_key]).strip():
            
            # Clean up the meanings (remove extra newlines, spaces)
            kannada = str(row[kannada_key]).strip().replace('\n', ' ')
            english = str(row[english_key]).strip().replace('\n', ' ')
            
            meanings[str(i)] = {
                "kannada": kannada,
                "english": english
            }
    return meanings

def process_sentences(meanings_file, sentences_file, api_key):
    # Add debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('classification_processing.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Read files with explicit UTF-8 encoding
    meanings_df = pd.read_csv(meanings_file, encoding='utf-8')
    sentences_df = pd.read_csv(sentences_file, encoding='utf-8')
    
    # Debug first few rows
    logging.debug(f"First row of meanings: {meanings_df.iloc[0].to_dict()}")
    
    total_words = len(meanings_df)
    completed_words = 0
    total_sentences = 0
    processed_sentences = 0
    
    classifier = SentenceClassifier(api_key)  # api_key is now a list
    batch_size = 15  # Reduced batch size
    min_wait_time = 5  # Minimum wait time between batches in seconds
    
    # Calculate total sentences
    for _, meaning_row in meanings_df.iterrows():
        word = meaning_row['word']
        word_sentences = sentences_df[sentences_df['word'] == word]['sentence'].tolist()
        total_sentences += len(word_sentences)
        processed_sentences += len(classifier.processed_data["processed_sentences"].get(word, []))
    
    logging.info(f"Total progress: {processed_sentences}/{total_sentences} sentences ({(processed_sentences/total_sentences)*100:.2f}%)")
    
    for _, meaning_row in meanings_df.iterrows():
        word = meaning_row['word']
        logging.info(f"Processing word: {word} ({completed_words + 1}/{total_words})")
        
        word_sentences = sentences_df[sentences_df['word'] == word]['sentence'].tolist()
        
        if not word_sentences:
            logging.warning(f"No sentences found for word: {word}")
            completed_words += 1
            continue
        
        # Skip already processed sentences
        processed_sentences = classifier.processed_data["processed_sentences"].get(word, [])
        remaining_sentences = [s for s in word_sentences if s not in processed_sentences]
        
        if not remaining_sentences:
            logging.info(f"All sentences for word '{word}' have been processed. Skipping.")
            completed_words += 1
            continue
            
        logging.info(f"Processing {len(remaining_sentences)}/{len(word_sentences)} remaining sentences for word '{word}'")
        meanings = create_meanings_dict(meaning_row)
        
        for i in range(0, len(remaining_sentences), batch_size):
            batch = remaining_sentences[i:i + batch_size]
            batch_index = i // batch_size
            
            logging.info(f"Processing batch {batch_index} ({len(batch)} sentences) for word {word}")
            try:
                classifier.classify_batch(word, meanings, batch, batch_index)
                logging.info(f"Waiting {min_wait_time} seconds before next batch...")
                time.sleep(min_wait_time)
            except Exception as e:
                logging.error(f"Failed to process batch after multiple retries: {str(e)}")
                logging.info("Taking a longer break before continuing...")
                time.sleep(300)  # 5 minute break
                continue
            
        completed_words += 1

if __name__ == "__main__":
    api_keys = [
        "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",
        "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"
    ]
    
    process_sentences(
        meanings_file="kanPolyMeanings.csv",
        sentences_file="kannada_polysemy_sentences.csv",
        api_key=api_keys  # Pass list of keys instead of single key
    )