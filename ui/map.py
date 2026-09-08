"""Island selection map using Folium."""

import html
import re

import folium
from streamlit_folium import st_folium

from config import ISLANDS


def _clean_island_name(value):
    """Normalize text returned by streamlit-folium to an exact island name."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("text")
    if value is None:
        return None

    text = html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()
    return text if text in ISLANDS else None


def _island_from_coordinates(clicked):
    """Resolve a clicked Folium marker from its latitude/longitude."""
    if not isinstance(clicked, dict):
        return None

    lat = clicked.get("lat")
    lon = clicked.get("lng", clicked.get("lon"))
    if lat is None or lon is None:
        return None

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None

    island, distance = min(
        (
            (name, (lat - island_lat) ** 2 + (lon - island_lon) ** 2)
            for name, (island_lat, island_lon) in ISLANDS.items()
        ),
        key=lambda item: item[1],
    )

    return island if distance < 0.0004 else None


def draw_island_map():
    """Draw the interactive map and return selected island name (or None)."""
    center = (28.5, -15.5)

    m = folium.Map(location=center, zoom_start=7, tiles=None)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors",
        name="OpenStreetMap",
        overlay=False,
        control=False,
    ).add_to(m)

    for isla, (lat, lon) in ISLANDS.items():
        folium.Marker(
            location=(lat, lon),
            tooltip=isla,
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        ).add_to(m)

    result = st_folium(m, key="map", width=800, height=600)
    if not result:
        return None

    selected = _island_from_coordinates(result.get("last_object_clicked"))
    if selected:
        return selected

    return _clean_island_name(
        result.get("last_object_clicked_tooltip")
        or result.get("last_object_clicked_popup")
    )
