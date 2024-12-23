# this app implements the user flow described in assets/day05.png

# 0. setup (i.e. train model on local compute)
    # todo - provide better UX whilst model trains - takes between 2 and 4 minutes, should only have to happen once until machine spins down.
    # one option is to host the model on S3
# 1. share location preferences
# 2. share interests
# 3. review recommendations
# 4. contact charity

import streamlit as st
from pathlib import Path
import pandas as pd
import os
import sys
import toml
import folium
from folium import plugins
from streamlit_folium import st_folium
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.similarity_manager import DocumentSimilarityManager
from src.data_processing import load_charity_data, prepare_text_field
from src.location_manager import LocationManager

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

@st.cache_resource
def get_location_manager():
    """Initialize location manager with API key"""
    secrets_path = Path(__file__).parent.parent / 'secrets.toml'
    secrets = toml.load(str(secrets_path))
    return LocationManager(secrets.get('OPENWEATHER_SECRET'))

def create_charity_map(filtered_df, user_lat, user_lon, distance):
    """Creates an interactive map showing charities and user location"""
    # Calculate center point
    center_lat = (filtered_df['latitude'].mean() + user_lat) / 2
    center_lon = (filtered_df['longitude'].mean() + user_lon) / 2
    
    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add user location
    folium.Marker(
        location=[user_lat, user_lon],
        popup='Your Location',
        icon=folium.Icon(color='red', icon='home', prefix='fa'),
        tooltip='You are here'
    ).add_to(m)
    
    # Add charity markers
    for _, row in filtered_df.iterrows():
        popup_content = f"""
        <div style="width: 200px">
            <b>{row['charity name']}</b><br>
            Program: {row['Program name']}<br>
            Distance: {row['distance']:.1f}km<br>
            <a href="{row['Charity weblink']}" target="_blank">Website</a>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='blue', icon='heart', prefix='fa'),
            tooltip=row['charity name']
        ).add_to(m)
    
    # Add search radius
    folium.Circle(
        location=[user_lat, user_lon],
        radius=distance * 1000,  # Convert km to meters
        color='red',
        fill=True,
        opacity=0.2,
        tooltip=f'{distance}km radius'
    ).add_to(m)
    
    # Add controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    
    return m

def main():
    st.title("Charity Finder")
    
    # Initialize session state
    if 'location_search_performed' not in st.session_state:
        st.session_state.location_search_performed = False
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'location_metadata' not in st.session_state:
        st.session_state.location_metadata = None
    
    # Initialize managers
    manager = get_similarity_manager()
    location_manager = get_location_manager()
    df = load_charity_data()
    
    # Create tabs for the different steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Location Preferences", 
        "2. Share Interests",
        "3. Review Recommendations",
        "4. Contact Charity"
    ])
    
    # Tab 1: Location Preferences
    with tab1:
        st.header("Find Charities Near You")
        
        # Get user inputs
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input(
                "Enter your location (City and State):",
                "Port Kembla, NSW",
                key="location_input"
            )
        with col2:
            distance = st.slider(
                "Maximum distance (km):",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                key="distance_slider"
            )
        
        if st.button("Find Nearby Charities", key="search_button"):
            try:
                # Process location query
                filtered_df, metadata = location_manager.process_location_query(
                    df, location, distance
                )
                
                # Update session state
                st.session_state.filtered_df = filtered_df
                st.session_state.location_metadata = metadata
                st.session_state.location_search_performed = True
                
                
            except Exception as e:
                st.error(f"Error processing location: {str(e)}")
        
        # Display results if search has been performed
        if st.session_state.location_search_performed:
            metadata = st.session_state.location_metadata
            filtered_df = st.session_state.filtered_df
            
            st.success(
                f"Found {metadata['num_charities']} charity programs within "
                f"{metadata['distance']}km of {metadata['location']}"
            )
            
            if not filtered_df.empty:
                # Create and display map
                m = create_charity_map(
                    filtered_df,
                    metadata['user_lat'],
                    metadata['user_lon'],
                    metadata['distance']
                )
                st_folium(m, width=800, height=600)
                
                # Display charity details
                st.subheader("Nearby Charity Programs (max 30):")
                for _, row in filtered_df.sort_values('distance').head(min(30,filtered_df.shape[0])).iterrows():
                    with st.expander(
                        f"{row['charity name']} - {row['Program name']} - ({row['distance']:.1f}km away)"
                    ):
                        st.write(f"**Program:** {row['Program name']}")
                        st.write(f"**Location:** {row['operating_location']}")
                        st.write(f"**Description:** {row['how purposes were pursued']}")
                        st.markdown(f"**Website:** [{row['Charity weblink']}]({row['Charity weblink']})")
    
    # Tab 2: Share Interests
    with tab2:
        st.header("What Causes Interest You?")
        query = st.text_input(
            "Describe the type of charity you're looking for:",
            "A soup kitchen for homeless people"
        )
        
        if st.button("Search"):
            similar_indices, similarities = manager.find_similar(query)
            
            if similar_indices:
                st.subheader("Most Similar Charities:")
                
                # Filter by location if available
                display_df = df
                if st.session_state.filtered_df is not None:
                    display_df = st.session_state.filtered_df
                
                # Display results
                for idx, sim in zip(similar_indices, similarities):
                    if idx in display_df.index:
                        with st.expander(f"{display_df.loc[idx, 'charity name']} - Similarity: {sim:.2f}"):
                            st.write(f"Program: {display_df.loc[idx, 'Program name']}")
                            st.write(f"Description: {display_df.loc[idx, 'how purposes were pursued']}")
                            st.write(f"Location: {display_df.loc[idx, 'operating_location']}")
            else:
                st.warning("No similar charities found.")
    
    # Placeholder for future tabs
    with tab3:
        st.header("Review Your Recommendations")
        st.info("This feature is coming soon!")
    
    with tab4:
        st.header("Contact Charity")
        st.info("This feature is coming soon!")

if __name__ == "__main__":
    main()