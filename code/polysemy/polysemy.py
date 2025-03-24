import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('data.csv')

# Create a new DataFrame for the polysemy data
polysemy_data = []

# Define the Kannada digits
kannada_digits = ['೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', '೧೦', '೧೧', '೧೨', '೧೩', '೧೪', '೧೫', '೧೬', '೧೭', '೧೮', '೧೯', '೨೦']

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    # Split the meanings based on the Kannada digits
    meanings = re.split('|'.join(map(re.escape, kannada_digits)), row['meaning'])
    
    # Only include words that have more than one meaning
    if len(meanings) > 2:
        # Iterate over the split meanings
        for i, meaning in enumerate(meanings):
            # If the meaning is not empty, add it to the polysemy data
            if meaning.strip():
                # Remove brackets from the meaning
                meaning = meaning.replace('(', '').replace(')', '')
                if i == 0:
                    origin = meaning
                else:
                    polysemy_data.append({
                        'word': row['word'],
                        'wordtype': row['wordtype'],
                        'origin': origin,
                        'meaning_number': i,
                        'meaning': meaning.strip()
                    })

# Create a DataFrame from the polysemy data
polysemy_df = pd.DataFrame(polysemy_data)

# Save the DataFrame to a CSV file
polysemy_df.to_csv('polysemy.csv', index=False)