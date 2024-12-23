# this app implements the user flow described in assets/day05.png

# 0. setup (i.e. train model on local compute)
# 1. share location preferences
# 2. share interests
# 3. review recommendations
# 4. contact charity


# 0. setup (i.e. train model on local compute)

import streamlit as st
from pathlib import Path
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.similarity_manager import DocumentSimilarityManager
from src.data_processing import load_charity_data, prepare_text_field

# Initialize similarity manager
@st.cache_resource
def get_similarity_manager():
    manager = DocumentSimilarityManager(
        model_dir="data/models",
        processed_dir="data/processed",
        debug=False
    )
    
    # Try to load existing model
    if not manager.load_model():
        st.info("Training new model...")
        # Load and process data
        df = load_charity_data()
        df = prepare_text_field(df)
        
        # Train model
        manager.train_model(df, text_column='text_to_embed')
        st.success("Model trained successfully!")
    
    return manager

def main():
    st.title("Charity Similarity Search")
    
    # Initialize similarity manager
    manager = get_similarity_manager()
    
    # Search interface
    query = st.text_input(
        "Describe the type of charity you're looking for:",
        "A soup kitchen for homeless people"
    )
    
    if st.button("Search"):
        similar_indices, similarities = manager.find_similar(query)
        
        if similar_indices:
            st.subheader("Most Similar Charities:")
            
            # Load original data for display
            df = load_charity_data()
            
            for idx, sim in zip(similar_indices, similarities):
                with st.expander(f"{df.loc[idx, 'charity name']} - Similarity: {sim:.2f}"):
                    st.write(f"Program: {df.loc[idx, 'Program name']}")
                    st.write(f"Description: {df.loc[idx, 'how purposes were pursued']}")
                    st.write(f"Location: {df.loc[idx, 'operating_location']}")
        else:
            st.warning("No similar charities found.")

if __name__ == "__main__":
    main()