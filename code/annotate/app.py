# app.py
import streamlit as st
import pandas as pd
from datetime import datetime

def load_data():
    df = pd.read_csv('polysemy_dataset_with_meanings.csv')
    return df

def main():
    st.title("Kannada Word Sense Annotation Verification")
    
    # Load data
    df = load_data()
    
    # Simple user identification
    annotator_name = st.text_input("Enter your name:", key="annotator")
    if not annotator_name:
        st.warning("Please enter your name to start verifying")
        return

    # Get unique words
    words = df['word'].unique()
    
    # Word selection
    selected_word = st.selectbox("Select word to verify:", words)
    
    # Get data for selected word
    word_data = df[df['word'] == selected_word]
    
    # Get unique senses for this word
    senses = word_data[['kannada_meaning', 'english_meaning']].drop_duplicates()
    
    # Filter out senses with no sentences
    valid_senses = []
    for _, sense_row in senses.iterrows():
        sense_count = len(word_data[
            (word_data['kannada_meaning'] == sense_row['kannada_meaning']) & 
            (word_data['english_meaning'] == sense_row['english_meaning'])
        ])
        if sense_count > 0:
            valid_senses.append((sense_row['kannada_meaning'], sense_row['english_meaning'], sense_count))
    
    if not valid_senses:
        st.warning("No sentences found for this word.")
        return
    
    # Create tabs for valid senses only
    sense_tabs = st.tabs([
        f"{kannada} ({english}) - {count} sentences"
        for kannada, english, count in valid_senses
    ])
    
    # Process each sense in its own tab
    for tab_idx, (tab, (kannada, english, _)) in enumerate(zip(sense_tabs, valid_senses)):
        with tab:
            sense_data = word_data[
                (word_data['kannada_meaning'] == kannada) & 
                (word_data['english_meaning'] == english)
            ]
            
            st.write(f"### Verifying sentences for sense: {kannada} ({english})")
            st.write(f"Total sentences to verify: {len(sense_data)}")
            
            # Store verification results for this sense
            verified_data = []
            verified_count = 0
            
            for idx, row in sense_data.iterrows():
                with st.expander(f"Sentence {idx + 1}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Sentence:**")
                        st.write(row['sentence'])
                    
                    with col2:
                        verification = st.radio(
                            "Is this sense correct?",
                            options=["", "Yes", "No"],
                            key=f"verify_{tab_idx}_{idx}"
                        )
                        
                        if verification:  # Only count if user has made a choice
                            verified_count += 1
                    
                    if verification:  # Only add to verified data if user has made a choice
                        verified_data.append({
                            'word': row['word'],
                            'sentence': row['sentence'],
                            'kannada_meaning': kannada,
                            'english_meaning': english,
                            'is_correct': verification,
                            'annotator': annotator_name,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
            
            # Show progress only if there are sentences
            if len(sense_data) > 0:
                st.progress(verified_count / len(sense_data))
                st.write(f"Verified: {verified_count}/{len(sense_data)} sentences")
            
            # Download button for this sense
            if verified_data:
                verified_df = pd.DataFrame(verified_data)
                csv = verified_df.to_csv(index=False)
                st.download_button(
                    label=f"Download Verifications for {kannada}",
                    data=csv,
                    file_name=f'{selected_word}_{kannada}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key=f"download_{tab_idx}"
                )

if __name__ == "__main__":
    main()