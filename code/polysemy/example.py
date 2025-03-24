from easygoogletranslate import EasyGoogleTranslate
import pandas as pd

translator = EasyGoogleTranslate(
    source_language='kn',
    target_language='en',
    timeout=10
)

df = pd.read_csv('polysemy.csv')
df = df.meaning

english = []

for i, word in enumerate(df):
    result = translator.translate(word)
    english.append({
        'english': result
    })

    # If i is a multiple of 100, save the DataFrame to a CSV file
    if (i + 1) % 100 == 0:
        english_df = pd.DataFrame(english)
        english_df.to_csv('english.csv', index=False)

# Save any remaining translations
english_df = pd.DataFrame(english)
english_df.to_csv('english.csv', index=False)