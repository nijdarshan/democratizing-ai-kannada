import anthropic
import pandas as pd
import json
import logging
import os
import time
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.model_selection import KFold

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
        
        # Update system message for the new task
        self.system_message = "You are an expert computational linguist specializing in word sense disambiguation. Respond only with the requested JSON format containing numeric labels."
        
        # Remove the old examples as we'll use k-fold examples instead
        self.examples = ""
        
        # Update instruction template to include training examples
        self.instruction_template = '''
        Analyze the following sentences and determine which sense number (1 or 2) applies to each test sentence.
        
        Training Examples:
        {training_examples}
        
        Guidelines:
        1. Use ONLY sense numbers 1 or 2
        2. Consider the complete context of each sentence
        3. Return ONLY the numeric labels in the specified JSON format
        4. Sense 1: Physical kiss/touching with lips
        5. Sense 2: Pearl/gem or metaphorical usage
        
        Test Sentences to classify:
        {sentences}

        Expected response format:
        {{
            "labels": [1, 2, 1]  // one number per test sentence: 1 for physical kiss, 2 for pearl/metaphor
        }}

        Return ONLY the JSON with numeric labels. No explanation needed.
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

    def format_training_examples(self, training_data):
        """Format training examples for the prompt"""
        formatted = ""
        for idx, row in training_data.iterrows():
            formatted += f"Sentence: {row['sentence']}\n"
            formatted += f"Label: {row['llm_annotation']}\n\n"
        return formatted

    def create_message(self, train_examples, test_sentences):
        """Create the message for the API"""
        formatted_examples = self.format_training_examples(train_examples)
        formatted_sentences = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(test_sentences))
        
        return {
            "system": self.system_message,
            "messages": [{
                "role": "user",
                "content": self.instruction_template.format(
                    training_examples=formatted_examples,
                    sentences=formatted_sentences
                )
            }]
        }

    @retry(
        wait=wait_exponential(multiplier=5, min=5, max=300),  # Reduced wait times
        stop=stop_after_attempt(5)
    )
    def classify_sentences(self, train_examples, test_sentences):
        """Classify a batch of sentences using the API"""
        message = self.create_message(train_examples, test_sentences)
        
        # Try each API key until one works
        for api_key in self.api_keys:
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json=message
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Parse the JSON response from the content
                    labels_json = json.loads(result['content'][0]['text'])
                    return labels_json
                    
            except Exception as e:
                logging.error(f"API error with key {api_key[:8]}...: {str(e)}")
                continue
                
        raise Exception("All API keys failed")

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

def process_sentences(sentences_file, api_keys):
    """Process sentences using k-fold cross validation"""
    
    # Load the dataset
    df = pd.read_csv(sentences_file, encoding='utf-8')
    df = df.sample(60, random_state=0)
    
    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    classifier = SentenceClassifier(api_keys)
    
    # Create a directory for storing k-fold results
    os.makedirs("kfold_results", exist_ok=True)
    
    # For each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        logging.info(f"Processing fold {fold_idx}")
        
        # Split data into train and test
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        try:
            # Process in batches
            batch_size = 5
            for i in range(0, len(test_data), batch_size):
                batch = test_data.iloc[i:i+batch_size]
                
                response = classifier.classify_sentences(
                    train_examples=train_data,
                    test_sentences=batch['sentence'].tolist()
                )
                
                # Store results for this batch
                results = {
                    "fold": fold_idx,
                    "batch": i // batch_size + 1,
                    "classifications": [
                        {
                            "sentence": sent,
                            "true_label": true,
                            "predicted_label": pred,
                            "word": word
                        }
                        for sent, true, pred, word in zip(
                            batch['sentence'],
                            batch['llm_annotation'],
                            response['labels'],
                            batch['word']
                        )
                    ],
                    "metadata": {
                        "training_examples_count": len(train_data),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": "claude-3-5-sonnet"
                    }
                }
                
                # Save results for this batch in fold-specific file
                results_file = f"kfold_results/fold_{fold_idx}_results.jsonl"
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(results, ensure_ascii=False) + "\n")
                
                # Also save accuracy summary
                accuracy = sum(1 for c in results["classifications"] 
                             if c["true_label"] == c["predicted_label"]) / len(results["classifications"])
                
                with open(f"kfold_results/fold_{fold_idx}_summary.txt", "a", encoding="utf-8") as f:
                    f.write(f"Batch {results['batch']}: Accuracy = {accuracy:.2%}\n")
                
                logging.info(f"Fold {fold_idx}, Batch {results['batch']}: Accuracy = {accuracy:.2%}")
                
                time.sleep(5)  # Rate limiting
                
        except Exception as e:
            logging.error(f"Error processing fold {fold_idx}: {str(e)}")
            continue


if __name__ == "__main__":
    api_keys = [
        "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",
        "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"
    ]
    
    process_sentences(
        sentences_file="all_predictions.csv",
        api_keys=api_keys
    )