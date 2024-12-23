# src/data_processing.py

import pandas as pd
from pathlib import Path
import logging

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent

def load_charity_data() -> pd.DataFrame:
    """Load and preprocess charity data with caching"""
    # Setup paths
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    processed_file = processed_dir / "processed_charities.pkl"
    
    # Check if processed data exists
    if processed_file.exists():
        logging.info("Loading preprocessed charity data from cache")
        return pd.read_pickle(processed_file)
    
    logging.info("Processing charity data from raw files")
    
    # Load raw data
    data_dir = project_root / "data" / "raw"
    charities_df = pd.read_csv(data_dir / "datadotgov_ais22.csv")
    programs_df = pd.read_csv(data_dir / "datadotgov_ais22_programs.csv")

    # 1. Filter for non-null websites in both dataframes
    programs_filtered = programs_df[programs_df['Charity weblink'].notna()].copy()
    charities_filtered = charities_df[charities_df['charity website'].notna()].copy()
    
    # 2. Join the dataframes on ABN/abn
    merged_df = pd.merge(
        charities_filtered,
        programs_filtered,
        left_on='abn',
        right_on='ABN',
        how='inner'
    )
    
    # 3. Create lists of location columns and their corresponding lat/long columns
    location_cols = [f'Operating Location {i}' for i in range(1, 11)]
    latlong_cols = [f'Operating Location {i} lat/long' for i in range(1, 11)]
    
    # 4. Melt the location columns
    locations_melted = pd.melt(
        merged_df,
        id_vars=[col for col in merged_df.columns if col not in location_cols + latlong_cols],
        value_vars=location_cols,
        var_name='location_number',
        value_name='operating_location'
    )
    
    # 5. Melt the lat/long columns
    latlong_melted = pd.melt(
        merged_df,
        value_vars=latlong_cols,
        var_name='latlong_number',
        value_name='lat_long'
    )
    
    # 6. Combine the melted dataframes
    final_df = locations_melted.copy()
    final_df['lat_long'] = latlong_melted['lat_long']
    
    # 7. Split lat/long into separate columns
    lat_lon_split = final_df['lat_long'].str.split('|', expand=True)
    final_df['latitude'] = pd.to_numeric(lat_lon_split[0], errors='coerce')
    final_df['longitude'] = pd.to_numeric(lat_lon_split[1], errors='coerce')
    
    # 8. Drop rows where lat/long are zero or null
    final_df = final_df.dropna(subset=['latitude', 'longitude'])
    final_df = final_df[
        (final_df['latitude'] != 0) & 
        (final_df['longitude'] != 0)
    ]
    
    # 9. Extract location number using string slicing instead of regex
    final_df['location_number'] = final_df['location_number'].str.slice(-2).str.strip().astype(int)
    
    # 10. Drop the original lat_long column as we now have split columns
    final_df = final_df.drop('lat_long', axis=1)
    
    # 11. Sort the dataframe by ABN and location number
    final_df = final_df.sort_values(['ABN', 'location_number'])
    
    # 12. Add a unique identifier
    final_df['id'] = final_df['charity name'].astype(str)+' | '+final_df['Program name']+' | '+final_df['operating_location'].astype(str)

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    final_df.to_pickle(processed_file)
    logging.info(f"Saved processed data to {processed_file}")
    
    return final_df

def prepare_text_field(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare text field for embedding"""
    
    # Combine relevant fields for text embedding
    df['text_to_embed'] = df['how purposes were pursued'].fillna('') + ' ' + \
                         df['Program name'].fillna('')
    # todo - add in more metadata fields

    return df