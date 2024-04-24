import pickle
import streamlit as st
import pandas as pd
import analysis
import forecast

# Sidebar navigation
def sidebar_navigation():
    
    # Add options for subpages
    selected_page = st.sidebar.radio("Wybierz zakres:", ("Analiza danych", "Prognoza sprzedaży"))
    
    # Display selected page content
    if selected_page == "Analiza danych":
        analysis.main()
    elif selected_page == "Prognoza sprzedaży":
        forecast.main()

# Run the app
if __name__ == "__main__":
    st.set_page_config(
        page_title="Sell Classifier",
        page_icon="images/icone.png",
    )

    with st.sidebar:
        st.markdown("## Platforma do wizualizacji analizy danych i prognozy sprzedaży")
        
    sidebar_navigation()
