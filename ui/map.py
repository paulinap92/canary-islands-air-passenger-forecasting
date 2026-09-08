"""Island selection map using Folium."""

import html
import re

import folium
from streamlit_folium import st_folium

from config import ISLANDS


def _clean_island_name(value):
    """Normalize map click output to an exact island name from ISLANDS."""
    if value is None:
        return None

    if isinstance(value, dict):
        value = value.get("text")

    if value is None:
        return None

    text = html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()
    return text if text in ISLANDS else None


def draw_island_map():
    """Draw the interactive map and return selected island name (or None)."""
    center = (28.5, -15.5)
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")

    for isla, (lat, lon) in ISLANDS.items():
        folium.Marker(
            location=(lat, lon),
            tooltip=isla,
            popup=folium.Popup(isla, max_width=150),
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        ).add_to(m)

    result = st_folium(m, key="map", width=800, height=600)
    if not result:
        return None

    selected = (
        result.get("last_object_clicked_tooltip")
        or result.get("last_object_clicked_popup")
    )
    return _clean_island_name(selected)
