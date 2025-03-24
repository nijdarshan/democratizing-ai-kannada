import anthropic
import pandas as pd
import json
import logging
import os
import time
from collections import defaultdict

class SentenceGenerator:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        # Cache structure: self.cache[meaning_key] = [sentences]
        # where meaning_key is a combination of kannada and english meaning
        self.cache = defaultdict(list)
        
    def _get_meaning_key(self, meaning):
        """Create a unique key for caching based on the meaning"""
        return f"{meaning['kannada']}|||{meaning['english']}"
        
    def generate_batch(self, word, meaning, batch_size, previous_sentences=None):
        try:
            # Check cache for this specific meaning
            meaning_key = self._get_meaning_key(meaning)
            cached_sentences = self.cache[meaning_key]
            if cached_sentences:
                logging.info(f"Found {len(cached_sentences)} cached sentences for meaning: {meaning['kannada']}")
                return cached_sentences

            message_data = self.create_message(word, meaning, batch_size, previous_sentences)
            
            # API call with retry logic
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=5000,
                        temperature=0.7,
                        **message_data
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)
            
            # Parse response and extract sentences
            response_data = json.loads(response.content[0].text)
            new_sentences = response_data.get("sentences", [])
            
            # Cache the new sentences for this meaning
            self.cache[meaning_key].extend(new_sentences)
            
            return new_sentences
            
        except Exception as e:
            logging.error(f"Error generating batch: {str(e)}")
            return []

    def generate_sentences(self, word, meaning, total_sentences):
        all_sentences = []
        remaining_sentences = total_sentences
        batch_size = min(10, total_sentences)  # Use smaller batches for efficiency
        
        # First check if we have enough cached sentences for this meaning
        meaning_key = self._get_meaning_key(meaning)
        cached_sentences = self.cache[meaning_key]
        if cached_sentences:
            if len(cached_sentences) >= total_sentences:
                logging.info(f"Using {total_sentences} cached sentences for meaning: {meaning['kannada']}")
                return cached_sentences[:total_sentences]
            else:
                all_sentences.extend(cached_sentences)
                remaining_sentences -= len(cached_sentences)
                logging.info(f"Using {len(cached_sentences)} cached sentences, generating {remaining_sentences} more")
        
        while remaining_sentences > 0:
            current_batch_size = min(batch_size, remaining_sentences)
            logging.info(f"Generating batch of {current_batch_size} sentences for meaning: {meaning['kannada']}")
            
            new_sentences = self.generate_batch(word, meaning, current_batch_size, all_sentences)
            all_sentences.extend(new_sentences)
            
            remaining_sentences -= len(new_sentences)
            time.sleep(1)  # Small delay between batches
        
        return all_sentences

    def create_message(self, word, meaning, batch_size, previous_sentences=None):
        system_message = """
        As a Kannada language expert, generate diverse and creative sentences using the given Kannada word.
        Focus on linguistic variety and natural usage while maintaining grammatical correctness.
        """

        examples = """
        <examples>
          <example>
            <input>
            {
              "word": "ಅಡಿ",
              "meaning": {
                "kannada": "ಅಳತೆಯ ಮಾನ",
                "english": "foot (measurement)"
              }
            }
            </input>
            <output>
            {
              "sentences": [
                "ಮೈಸೂರು ಅರಮನೆ ನೂರು ಅಡಿ ಎತ್ತರ",
                "ಅವಳ ತೋಟದಲ್ಲಿ ಐವತ್ತು ಅಡಿ ಆಳದ ಬಾವಿ ಇದೆ"
              ]
            }
            </output>
          </example>
        </examples>
        """

        # Create avoid list from previous sentences
        avoid_list = ""
        if previous_sentences:
            avoid_list = "\nAvoid generating sentences similar to these previously generated ones:\n"
            avoid_list += "\n".join(f"- {s}" for s in previous_sentences)

        instructions = f"""
        Generate {batch_size} new sentences for:
        Word: {word}
        Meaning (Kannada): {meaning['kannada']}
        Meaning (English): {meaning['english']}

        Linguistic Requirements:
        1. Use ONLY the base form '{word}' - STRICTLY no modifications or suffixes
        2. The word to be in different positions of the sentence (early, mid, late)
        3. Create sentences with varying lengths (short, medium, long)
        4. Include a mix of:
           - Different pronouns (ನಾನು, ಅವನು, ಅವಳು, ಅವರು, etc.)
           - Various adjectives (qualities, descriptions)
           - Adverbs (time, manner, place)
           - Different tenses and aspects
           - Questions and statements
           - Direct and indirect speech
        5. Incorporate:
           - Modern and traditional contexts
           - Urban and rural settings
           - Professional and casual situations
           - Different age groups and social roles
        6. Use diverse proper nouns:
           - People names (both Indian and international)
           - Place names (cities, villages, landmarks)
           - Organization names
        7. Vary sentence structure:
           - Simple sentences
           - Compound sentences
           - Complex sentences with subordinate clauses
        8. The word might have synonyms as meanings and might be explicit, so understand the meaning beore making sentences
        
        {avoid_list}

        Return ONLY JSON with the new sentences:
        {{
          "sentences": [
            "sentence1",
            "sentence2",
            ...
          ]
        }}
        """

        return {
            "system": system_message,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": examples}]},
                {"role": "user", "content": [{"type": "text", "text": instructions}]}
            ]
        }

    def generate_batch(self, word, meaning, batch_size, previous_sentences=None):
        try:
            message_data = self.create_message(word, meaning, batch_size, previous_sentences)
            
            # API call with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(message_data)
                    response = ""
                    # response = self.client.messages.create(
                    #     model="claude-3-5-sonnet-20241022",
                    #     max_tokens=5000,
                    #     temperature=0.7,
                    #     **message_data
                    # )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)
            
            # Parse response and extract sentences
            response_data = json.loads(response.content[0].text)
            return response_data.get("sentences", [])
            
        except Exception as e:
            logging.error(f"Error generating batch: {str(e)}")
            return []

    def generate_sentences(self, word, meaning, total_sentences):
        all_sentences = []
        remaining_sentences = total_sentences
        batch_size = min(10, total_sentences)  # Use smaller batches for efficiency
        
        while remaining_sentences > 0:
            current_batch_size = min(batch_size, remaining_sentences)
            logging.info(f"Generating batch of {current_batch_size} sentences for {word} ({meaning['kannada']})")
            
            new_sentences = self.generate_batch(word, meaning, current_batch_size, all_sentences)
            all_sentences.extend(new_sentences)
            
            remaining_sentences -= len(new_sentences)
            time.sleep(1)  # Small delay between batches
        
        return all_sentences

def create_meanings_dict(row):
    """Create meanings dictionary from row data"""
    meanings = []
    for i in range(1, 5):  # Max 4 meanings
        kannada_key = f'kannada_meaning_{i}'
        english_key = f'english_meaning_{i}'
        
        if pd.notna(row.get(kannada_key)) and pd.notna(row.get(english_key)):
            meanings.append({
                "kannada": row[kannada_key],
                "english": row[english_key],
                "index": i
            })
            print(meanings)
    return meanings

def process_words(csv_file, api_key, output_dir="output"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'processing.log')),
            logging.StreamHandler()
        ]
    )

    df = pd.read_csv(csv_file)
    #df = df.sample(3, random_state=22)
    generator = SentenceGenerator(api_key)
    results = []

    for _, row in df.iterrows():
        word = row['word']
        meanings = create_meanings_dict(row)
        
        word_results = {
            "word": word,
            "meanings": []
        }

        for meaning in meanings:
            logging.info(f"Processing word: {word} - meaning {meaning['index']}")
            
            # Determine number of sentences based on meaning index
            sentences_to_generate = 50 if meaning['index'] <= 2 else 20
            
            sentences = generator.generate_sentences(word, meaning, sentences_to_generate)
            
            word_results["meanings"].append({
                "meaning_index": meaning['index'],
                "definition": meaning['kannada'],
                "english": meaning['english'],
                "sentences": sentences
            })

            # Save intermediate results
            save_results(word_results, f"{word}_intermediate.json", output_dir)
        
        results.append(word_results)
        save_results(results, "all_sentences.json", output_dir)

def save_results(data, filename, output_dir):
    try:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved results to {filepath}")
    except Exception as e:
        logging.error(f"Error saving results to {filename}: {str(e)}")

if __name__ == "__main__":
    api_key = ""  # Add your API key here
    process_words("kanPolyMeanings.csv", "")