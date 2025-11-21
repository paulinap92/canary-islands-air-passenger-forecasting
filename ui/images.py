"""Display island image next to the map."""

import streamlit as st
from config import ISLAND_IMAGES

def show_island_image(selected_island: str) -> None:
    """Show image for a selected island using Pexels URL."""
    if not selected_island:
        st.info("Haz click en una isla del mapa para ver su imagen.")
        return

    img_url = ISLAND_IMAGES.get(selected_island)
    if img_url:
        st.markdown(
            f"<div class='fixed-image'><img src='{img_url}' style='width:100%; border-radius:10px;'></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"📸 {selected_island}")
    else:
        st.info(f"No hay imagen disponible para {selected_island}.")

def show_image_license():
    """Show a small license / credit note for Pexels images."""
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:13px; margin-top:-10px;'>"
        "📸 Fotos: Pexels — uso libre; se recomienda atribución a los autores originales.</div>",
        unsafe_allow_html=True,
    )
