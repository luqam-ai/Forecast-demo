import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis import data_for_product
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
import json

def train_prophet_model(df, prod):
    prod_df = data_for_product(df, prod)
    df = prod_df[['Date', 'Sale_quantity_final', 'Mieszkania', 'CPI', 'Wsk_bud_mont', 'Sum_q_other_lamels_by_month', 'Promo_ind', 'Promo_ind_all']]
    df.rename(columns={"Date": "ds", "Sale_quantity_final": "y"}, inplace=True)
    all_data = df

    model = Prophet()
    model.add_regressor('Mieszkania')
    model.add_regressor('CPI')
    model.add_regressor('Wsk_bud_mont')
    # model.add_regressor('Sum_q_other_lamels_by_month')
    model.add_regressor('Promo_ind')
    #model.add_regressor('Promo_ind_all')
    model.fit(all_data)
    return all_data, model

def forecast_next_month(df, model, mieszkania, wsk_bm, cpi, promo, next_date='20230701'):
    d = pd.to_datetime(next_date, format='%Y%m%d', errors='ignore')
    data = {"ds": pd.to_datetime('20230701', format='%Y%m%d', errors='ignore')}
    future = pd.DataFrame(data, index=[0])
    future['Mieszkania'] = mieszkania
    future['CPI'] = cpi
    future['Wsk_bud_mont'] = wsk_bm
    future['Promo_ind'] = promo
    future = df.append(future)
    forecast = model.predict(future)
    return forecast

def plot_forecast_vs_real(df, forecast, prod):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.ds,
                    y=df.y,
                    name=f'Historyczne dane sprzedażowe',
                    marker_color='rgb(55, 83, 109)',
                    # mode='lines',
                    ))
    fig.add_trace(go.Bar(x=forecast.iloc[-1:].ds,
                    y=forecast.iloc[-1:].yhat,
                    name='Prognoza sprzedaży',
                    #mode='lines',
                    marker_color='rgb(26, 118, 255)'
                    ))

    fig.update_layout(
        title=f'Sprzedaż {prod}',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Liczba sztuk',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
    )
    return fig

def main():
        
    # Page title
    st.title('Pognozowanie sprzedaży')
    st.image('images/prognozowanie-sprzedazy.png')
    st.write("\n\n")
    
    df = st.session_state['df']
    prod = st.selectbox(
        'Wybierz produkt:',
        list(df.Prod_code.unique()))
    st.session_state['prod'] = prod
    
    st.subheader("Symulacja wskaźników na kolejny miesiąc")
    st.markdown("#### Lipiec 2023:")
    cpi = st.slider('Wskaźnik wzrostu cen (inflacja)', min_value=70.0, max_value=120.0, value=102.6, step=0.1)
    wsk_bm = st.slider('Wskaźnik cen budowlano-montażowych', min_value=70.0, max_value=120.0, value=101.5, step=0.1)
    mieszkania = st.slider('Liczba mieszkań oddanych do użytkowania', min_value=15000, max_value=30000, value=17600, step=1)
    promo = st.slider('Wskaźnik promocji (liczba punktów, w których ma wystąpić promocja na produkt)', min_value=0, max_value=len(df.Receiver_code.unique()), value=12, step=1)
    
    st.subheader(f"Prognoza sprzedaży produktu na kolejny miesiąc")
    st.markdown("#### Lipiec 2023:")
    forecast_btn = st.button('Prognoza')
    
    if forecast_btn:
        all_data, model = train_prophet_model(df, prod)
        forecast = forecast_next_month(all_data, model, mieszkania, wsk_bm, cpi, promo)

        f, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(all_data.ds, all_data['y'], color='r')
        fig = model.plot(forecast)
        st.pyplot(fig)
        file = f"{prod}_mieszk-{mieszkania}_cpi-{cpi}_wskbm-{wsk_bm}_promo-{promo}"
        print(f"Write to file {file}")
        fig.savefig(f'{file}_model.png')

        fig_real_forecast = plot_forecast_vs_real(all_data, forecast, prod)
        fig_real_forecast.write_json(f"{file}_real_forecast.json")
        st.plotly_chart(fig_real_forecast)




if __name__ == "__main__":
    main()