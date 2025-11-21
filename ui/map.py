"""Island selection map using Folium."""

import folium
from streamlit_folium import st_folium
import streamlit as st
from config import ISLANDS

def draw_island_map():
    """Draw the interactive map and return selected island name (or None)."""
    center = (28.5, -15.5)
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    for isla, (lat, lon) in ISLANDS.items():
        folium.Marker(
            location=(lat, lon),
            tooltip=isla,
            popup=folium.Popup(f"<b>{isla}</b>", max_width=150),
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        ).add_to(m)

    result = st_folium(m, key="map", width=800, height=520)
    clicked_popup = result.get("last_object_clicked_popup") if result else None

    if clicked_popup:
        sel = str(clicked_popup.get("text") if isinstance(clicked_popup, dict) else clicked_popup)
        return sel
    return None
