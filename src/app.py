# this app implements the user flow described in assets/day05.png

# 0. setup (i.e. train model on local compute)
    # todo - provide better UX whilst model trains - takes between 2 and 4 minutes, should only have to happen once until machine spins down.
    # one option is to host the model on S3
# 1. share location preferences
    # todo - resolve map flickering
# 2. share interests
    # todo - add map -folium crashes things
# 3. AI recommendations
    # todo - add map. Add default searches.
# 4. contact charity
    # todo - add "was this helpful?"

import streamlit as st
from pathlib import Path
import pandas as pd
import os
import sys
import toml
import folium
from folium import plugins
from streamlit_folium import st_folium
import json
import aisuite as ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.similarity_manager import DocumentSimilarityManager
from src.data_processing import load_charity_data, prepare_text_field
from src.location_manager import LocationManager

# System prompt for AI recommendations
SYSTEM_PROMPT = """You are a charity volunteering recommendation assistant for Australian nonprofits. Your role is to thoughtfully match volunteers with charities based on their interests and location.

For each recommendation, consider:
1. How well the charity's activities match the user's interests (indicated by similarity score where 1.0 is perfect)
2. Geographic accessibility from the user's location (indicated by distance in km)

Your response must be a SINGLE valid JSON object with NO special characters or formatting beyond simple spaces and escaped newlines (\\n). Include these two keys:
- charity_ids: list of recommended charity IDs (maximum 10)
- message: single string with newlines represented as \\n

Example of valid response format:
{
    "charity_ids": [26164, 14931],
    "message": "I found two excellent matches for you!\\n\\nThe Wollongong Homeless Hub (2km away, 0.92 similarity) aligns perfectly with your interest in helping the homeless.\\n\\nNeed a Feed Australia (5km away, 0.87 similarity) also runs food relief programs and needs delivery drivers."
}

If no matches found:
{
    "charity_ids": [],
    "message": "I couldn't find exact matches in your area. The map below shows nearby charities you might consider."
}"""

def prepare_charity_context(similar_indices, similarities, df, max_chars=100000):
    """Prepare charity context for LLM prompt while respecting context length limits."""
    context_parts = []
    included_ids = []
    current_length = 0
    
    for idx, sim in zip(similar_indices, similarities):
        charity = df.loc[idx]
        
        entry = (
            f"ID: {idx}\n"
            f"Name: {charity['charity name'][:100]}\n"
            f"Program: {charity['Program name'][:100]}\n"
            f"Location: {charity['operating_location'][:150]}\n"
            f"Distance: {charity['distance']}\n"
            f"Description: {charity['how purposes were pursued'][:2000]}\n"
            f"Similarity: {sim:.3f}\n"
            f"Website: {charity.get('Charity weblink', 'Not available')}\n"
            "---\n"
        )
        
        if current_length + len(entry) > max_chars:
            break
            
        context_parts.append(entry)
        included_ids.append(idx)
        current_length += len(entry)
    
    context = (
        f"There are {len(included_ids)} relevant charities that may match this interest.\n"
        f"Similarity scores range from {similarities[0]:.3f} to {similarities[-1]:.3f}.\n"
        "Available charity details:\n\n"
        + "\n".join(context_parts)
    )
    
    return context, included_ids

@st.cache_resource
def get_ai_client():
    """Initialize AI client with API key"""
    secrets_path = Path(__file__).parent.parent / 'secrets.toml'
    secrets = toml.load(str(secrets_path))
    os.environ['ANTHROPIC_API_KEY'] = secrets.get('ANTHROPIC_SECRET')
    return ai.Client()

def get_ai_recommendations(client, location, query, context_text, valid_ids):
    """Get AI recommendations for charities"""
    user_prompt = f"""
    The user provided the following location: {location}.\n
    The user provided the following query: {query}.\n
    Use this context to recommend up to 10 charities: {context_text}
    Valid id's are as follows: {valid_ids}
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    response = client.chat.completions.create(
        model='anthropic:claude-3-5-haiku-20241022',
        messages=messages,
        temperature=0.75
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing AI response: {e}")
        return None

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
        # Create a more engaging loading message
        loading_container = st.empty()
        loading_container.info("""
        🤖 First-time setup in progress! We're training an AI model to understand charity descriptions...
        
        This takes about 5 minutes, so it's a perfect time to:
        - ☕ Grab a coffee or tea
        - 🫖 Put the kettle on
        - 🎵 Listen to your favorite song
        - 🤸 Do a quick stretch
        
        Don't worry - this only happens once! Next time you visit, it'll be instant.
        """)
        
        # Add a progress bar
        progress_bar = st.progress(0)
        
        # Load and process data with progress updates
        loading_container.info("📚 Loading charity data...")
        progress_bar.progress(25)
        df = load_charity_data()
        
        loading_container.info("🔍 Processing text descriptions...")
        progress_bar.progress(50)
        df = prepare_text_field(df)
        
        loading_container.info("🧠 Training embedding model...")
        progress_bar.progress(75)
        manager.train_model(df, text_column='text_to_embed')
        
        # Final success message
        progress_bar.progress(100)
        loading_container.success("✨ Setup complete! Let's find you the perfect volunteer opportunity!")
    
    return manager

@st.cache_resource
def get_location_manager():
    """Initialize location manager with API key"""
    secrets_path = Path(__file__).parent.parent / 'secrets.toml'
    secrets = toml.load(str(secrets_path))
    return LocationManager(secrets.get('OPENWEATHER_SECRET'))

def get_marker_color(similarity_score):
    """Returns marker color based on similarity score"""
    if similarity_score >= 0.5:
        return 'darkgreen'
    elif similarity_score >= 0.3:
        return 'green'
    elif similarity_score >= 0.2:
        return 'lightgreen'
    return 'blue'

def create_charity_map(filtered_df, user_lat, user_lon, distance, similarities_dict=None):
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
        similarity_score = similarities_dict.get(row.name, 0) if similarities_dict else 0
        marker_color = get_marker_color(similarity_score)
        
        popup_content = f"""
        <div style="width: 200px">
            <b>{row['charity name']}</b><br>
            Program: {row['Program name']}<br>
            Distance: {row['distance']:.1f}km<br>
            {'Similarity: {:.2f}<br>'.format(similarity_score) if similarities_dict else ''}
            <a href="{row['Charity weblink']}" target="_blank">Website</a>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=marker_color, icon='heart', prefix='fa'),
            tooltip=f"{row['charity name']} ({similarity_score:.2f})" if similarities_dict else row['charity name']
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
    
    # Add legend
    if similarities_dict:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 130px; 
                    border:2px solid grey; z-index:9999; background-color:white;
                    padding: 10px;
                    font-size: 14px;">
        <b>Similarity Score</b><br>
        <i class="fa fa-heart fa-1x" style="color:darkgreen"></i> &gt;= 0.5<br>
        <i class="fa fa-heart fa-1x" style="color:green"></i> &gt;= 0.3<br>
        <i class="fa fa-heart fa-1x" style="color:lightgreen"></i> &gt;= 0.2<br>
        <i class="fa fa-heart fa-1x" style="color:blue"></i> &lt; 0.2
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    
    return m

def format_charity_df(df):
    """Format charity dataframe for display"""
    display_cols = [
        'Charity weblink',        
        'distance',        
        'charity name', 
        'Program name',
        'operating_location',
        'total full time equivalent staff',
        'staff - volunteers',

    ]
    
    # Ensure all columns exist
    for col in display_cols:
        if col not in df.columns:
            df[col] = 'N/A'
    
    formatted_df = df[display_cols].copy()
    
    # Format numeric columns
    formatted_df['distance'] = formatted_df['distance'].apply(lambda x: f"{x:.1f}km" if pd.notnull(x) else "N/A")
    formatted_df['total full time equivalent staff'] = formatted_df['total full time equivalent staff'].fillna('N/A')
    formatted_df['staff - volunteers'] = formatted_df['staff - volunteers'].fillna('N/A')
    
    # Rename columns for display
    formatted_df.columns = [
        'Link',
        'Distance',        
        'Charity Name',
        'Program',
        'Location',
        'Full-time Staff',
        'Volunteers',
    ]
    
    return formatted_df

def display_charity_section(df, title, description):
    """Display a section of charities with formatting"""
    if df is not None and not df.empty:
        st.subheader(title)
        st.markdown(description)
        st.dataframe(
            format_charity_df(df),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn()
            }
        )
    else:
        st.info(f"No charities available for {title}")

def main():
    st.title("Oz Charity Finder")
    
    # Initialize session state
    if 'location_search_performed' not in st.session_state:
        st.session_state.location_search_performed = False
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'location_metadata' not in st.session_state:
        st.session_state.location_metadata = None
    if 'similarities_dict' not in st.session_state:
        st.session_state.similarities_dict = None
    if 'ai_recommendations' not in st.session_state:
        st.session_state.ai_recommendations = None
    if 'query' not in st.session_state:
        st.session_state.query = None        
    
    # Initialize managers
    manager = get_similarity_manager()
    location_manager = get_location_manager()
    ai_client = get_ai_client()
    df = load_charity_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "① Location Preferences", 
        "② Share Interests",
        "③ AI Recommendations",
        "④ Contact Charities",
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
                st.session_state.similarities_dict = None  # Reset similarities when location changes
                
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
        
        if not st.session_state.location_search_performed:
            st.warning("Please complete Step 1 (Location Preferences) first!")
            return
            
        query = st.text_input(
            "Describe the type of charity you're looking for:",
            "A soup kitchen for homeless people"
        )
        
        if st.button("Search"):
            # Store the query in session state
            st.session_state.query = query            
            if st.session_state.filtered_df is None:
                st.error("Please complete the location search first!")
                return
                
            filtered_df = st.session_state.filtered_df
            metadata = st.session_state.location_metadata
            
            # Find similar charities within the filtered set
            similar_indices, similarities = manager.find_similar_filtered(
                query=query,
                valid_indices=set(filtered_df.index),
                top_k=min(100, len(filtered_df))
            )
            
            # Debug information
            st.write("Number of similar charities found:", len(similar_indices))
            if similarities:
                st.write("Similarity score range:", f"{min(similarities):.2f} to {max(similarities):.2f}")
            
            if similar_indices:
                st.session_state.similarities_dict = dict(zip(similar_indices, similarities))
                
                # Display results first
                st.subheader("Similar Charity Programs:")
                for idx, sim in zip(similar_indices, similarities):
                    row = filtered_df.loc[idx]
                    with st.expander(
                        f"{row['charity name']} - {row['Program name']} - Similarity: {sim:.2f} - ({row['distance']:.1f}km away)",
                        expanded=False
                    ):
                        st.write(f"**Program:** {row['Program name']}")
                        st.write(f"**Location:** {row['operating_location']}")
                        st.write(f"**Description:** {row['how purposes were pursued']}")
                        st.markdown(f"**Website:** [{row['Charity weblink']}]({row['Charity weblink']})")
                
                # Commented out map for now while debugging
                # with st.container():
                #     st.subheader("Charity Locations:")
                #     m = create_charity_map(
                #         filtered_df,
                #         metadata['user_lat'],
                #         metadata['user_lon'],
                #         metadata['distance'],
                #         st.session_state.similarities_dict
                #     )
                #     st_folium(m, width=800, height=600)
            else:
                st.warning("No similar charities found in your area.")
    
    # Placeholder for future tabs
    # Tab 3: AI Recommendations
    with tab3:
        st.header("AI-Powered Recommendations")
        
        if not st.session_state.similarities_dict:
            st.warning("Please complete Step 2 (Share Interests) first!")
            return
            
        if st.button("Get AI Recommendations"):
            with st.spinner("Getting personalized recommendations..."):
                # Prepare context for AI
                context_text, valid_ids = prepare_charity_context(
                    list(st.session_state.similarities_dict.keys()),
                    list(st.session_state.similarities_dict.values()),
                    st.session_state.filtered_df
                )
                
                # Get AI recommendations
                recommendations = get_ai_recommendations(
                    ai_client,
                    st.session_state.location_metadata['location'],
                    st.session_state.query,
                    context_text,
                    valid_ids
                )
                
                if recommendations:
                    st.session_state.ai_recommendations = recommendations
                    
                    # Display AI message
                    st.markdown(recommendations['message'].replace('\\n', '\n'))
                    
                    # Filter and display recommended charities
                    recommended_df = st.session_state.filtered_df.loc[recommendations['charity_ids']]
                    
                    # # Create map with only recommended charities
                    # m = create_charity_map(
                    #     recommended_df,
                    #     st.session_state.location_metadata['user_lat'],
                    #     st.session_state.location_metadata['user_lon'],
                    #     st.session_state.location_metadata['distance'],
                    #     {idx: st.session_state.similarities_dict[idx] for idx in recommendations['charity_ids']}
                    # )
                    # st_folium(m, width=800, height=600)
                    
                    # Display detailed charity information
                    st.subheader("Recommended Charity Programs:")
                    for idx in recommendations['charity_ids']:
                        row = recommended_df.loc[idx]
                        similarity = st.session_state.similarities_dict[idx]
                        with st.expander(
                            f"{row['charity name']} - {row['Program name']} - Similarity: {similarity:.2f} - ({row['distance']:.1f}km away)",
                            expanded=False
                        ):
                            st.write(f"**Program:** {row['Program name']}")
                            st.write(f"**Location:** {row['operating_location']}")
                            st.write(f"**Description:** {row['how purposes were pursued']}")
                            st.markdown(f"**Website:** [{row['Charity weblink']}]({row['Charity weblink']})")

    with tab4:
        st.header("Contact Charities")
        
        if not st.session_state.get('ai_recommendations'):
            st.warning("Please complete Step 3 (AI Recommendations) first!")
            return
            
        # Get the filtered dataframe
        filtered_df = st.session_state.filtered_df
        
        if filtered_df is None or filtered_df.empty:
            st.error("No charity data available. Please complete the previous steps.")
            return
            
        # Create three dataframes for different recommendation types
        ai_recommended_df = None
        similar_df = None
        other_nearby_df = None
        
        # 1. AI Recommendations
        if st.session_state.ai_recommendations and 'charity_ids' in st.session_state.ai_recommendations:
            ai_ids = st.session_state.ai_recommendations['charity_ids']
            ai_recommended_df = filtered_df.loc[ai_ids]
            
            # Remove AI recommended charities from other sets
            remaining_df = filtered_df.drop(ai_ids, errors='ignore')
        else:
            remaining_df = filtered_df
            
        # 2. Similar Charities (high similarity but not AI recommended)
        if st.session_state.similarities_dict:
            # Get charities with similarity >= 0.2 that weren't AI recommended
            similar_indices = [
                idx for idx, sim in st.session_state.similarities_dict.items() 
                if sim >= 0.1 and idx in remaining_df.index
            ]
            similar_df = remaining_df.loc[similar_indices]
            
            # Remove similar charities from remaining set
            remaining_df = remaining_df.drop(similar_indices, errors='ignore')
        
        # 3. Other nearby charities
        other_nearby_df = remaining_df
        
        # Display sections
        st.markdown("### 📊 Summary of Available Charities")
        st.markdown("""
        Below you'll find three categories of charities:
        1. **AI Recommended**: Best matches based on your interests and location
        2. **Similar Programs**: Other programs that may match your interests
        3. **Nearby Charities**: Additional charities in your area
        """)
        
        # Display each section
        display_charity_section(
            ai_recommended_df,
            "🎯 AI Recommended Charities",
            "These charities best match your interests and location preferences:"
        )
        
        display_charity_section(
            similar_df,
            "👥 Similar Programs",
            "These programs also align with your interests:"
        )
        
        display_charity_section(
            other_nearby_df,
            "📍 Other Nearby Charities",
            "Additional charities in your area:"
        )
        
        # Add contact instructions
        st.markdown("""
        ### 📬 Next Steps
        
        To get involved with a charity:
        1. Click the website link to learn more about their programs
        2. Look for volunteer or contact information on their website
        3. Reach out directly to discuss volunteer opportunities
        
        Remember to mention your specific interests and availability when contacting them!
        """)


if __name__ == "__main__":
    main()