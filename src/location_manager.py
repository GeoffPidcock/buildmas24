# src/location_manager.py

import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, Dict

class LocationManager:
    def __init__(self, api_key: str):
        """Initialize LocationManager with OpenWeather API key"""
        self.api_key = api_key
    
    @staticmethod
    @st.cache_data
    def geocode_location(_api_key: str, location: str) -> Tuple[float, float]:
        """
        Convert location string to latitude and longitude using OpenWeather API
        """
        try:
            res = requests.get(
                f"http://api.openweathermap.org/geo/1.0/direct",
                params={
                    "q": f"{location}, Australia",
                    "limit": 1,
                    "appid": _api_key
                }
            )
            res.raise_for_status()
            location_result = res.json()
            
            if not location_result:
                raise ValueError(f"No results found for location: {location}")
                
            return location_result[0]['lat'], location_result[0]['lon']
            
        except requests.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            raise
    
    @staticmethod
    def haversine_distance(row: pd.Series, user_lat: float, user_lon: float) -> float:
        """Calculate Haversine distance between user location and charity location"""
        R = 6371  # Earth's radius in kilometers
            
        # Convert to radians
        user_lat_rad = np.radians(user_lat)
        user_lon_rad = np.radians(user_lon)
        charity_lats_rad = np.radians(row['latitude'])
        charity_lons_rad = np.radians(row['longitude'])
        
        # Haversine formula
        dlat = charity_lats_rad - user_lat_rad
        dlon = charity_lons_rad - user_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(user_lat_rad) * np.cos(charity_lats_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    @staticmethod
    @st.cache_data
    def filter_charities_by_location(df: pd.DataFrame, location: str, distance: float, 
                                   _api_key: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter charities based on location and distance
        """
        # Get coordinates
        user_lat, user_lon = LocationManager.geocode_location(_api_key, location)
        
        # Calculate distances
        df = df.copy()  # Create a copy to avoid modifying the original
        df['distance'] = df.apply(
            lambda row: LocationManager.haversine_distance(row, user_lat, user_lon), 
            axis=1
        )
        
        # Filter based on distance
        filtered_df = df.loc[df.distance <= distance].copy()
        
        # Prepare metadata
        metadata = {
            'location': location,
            'distance': distance,
            'user_lat': user_lat,
            'user_lon': user_lon,
            'num_charities': len(filtered_df)
        }
        
        return filtered_df, metadata

    def process_location_query(self, df: pd.DataFrame, location: str, 
                             distance: float) -> Tuple[pd.DataFrame, Dict]:
        """
        Process location query and return filtered results with metadata
        """
        return self.filter_charities_by_location(df, location, distance, self.api_key)