import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import json
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime

@st.experimental_memo
def read_data():
    path = "dataset/dataset1.csv"
    df = pd.read_csv(path, index_col=0)
    return df

@st.experimental_memo
def data_for_product(dataset, prod):
    dataset = dataset[['Year', 'Month', 'Date', 'Prod_code', 'Receiver_code',
       'Sale_quantity_final', 'Mieszkania', 'CPI',
       'Wsk_bud_mont', 'Promo', 'Start_date_for_rec']]
    dataset["Date"] = pd.to_datetime(dataset["Date"])

    #Sum lamels by month and Promo index by month for all lamels as features
    sum_lamels_by_month = dataset.groupby(["Date"])["Sale_quantity_final", "Promo"].agg("sum").reset_index()
    sum_lamels_by_month = sum_lamels_by_month.rename(columns={"Sale_quantity_final": "Sum_q_lamels_by_month", "Promo": "Promo_ind_all"})
    dataset_with_sums = dataset.merge(sum_lamels_by_month)
    dataset_with_sums["Sum_q_other_lamels_by_month"] = dataset_with_sums["Sum_q_lamels_by_month"] - dataset_with_sums["Sale_quantity_final"]

    #Add info about promotions by product
    promotions = dataset_with_sums.groupby(["Date", 'Prod_code'])['Promo'].agg(list).reset_index()
    promotions['Promo_ind'] = promotions['Promo'].apply(lambda promo_val: sum(promo_val))
    promotions = promotions.drop('Promo', axis=1)
    dataset_with_sums_and_promo_ind = dataset_with_sums.merge(promotions)

    #Sum quantity of sales of all receivers for chosen products
    dataset_all_receivers = dataset_with_sums_and_promo_ind.groupby(["Date","Year","Month", "Prod_code", "Mieszkania", "CPI", "Wsk_bud_mont", "Sum_q_lamels_by_month", "Promo_ind_all", "Promo_ind"])["Sale_quantity_final"].sum().reset_index()
    dataset_all_receivers["Sum_q_other_lamels_by_month"] = dataset_all_receivers["Sum_q_lamels_by_month"] - dataset_all_receivers["Sale_quantity_final"]

    #Select only one product
    prod1 = dataset_all_receivers[dataset_all_receivers.Prod_code==prod]
    return prod1


def plot_data_with_features(prod1, prod):
    #print(prod1.head)
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=prod1['Date'], y=prod1['CPI'], mode='lines', name='CPI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=prod1['Date'], y=prod1['Wsk_bud_mont'], mode='lines', name='Wskaźnik bud-mont'), row=1, col=1)
    fig.add_trace(go.Bar(x=prod1['Date'], y=prod1['Mieszkania'], name='Mieszkania'), row=2, col=1)
    fig.add_trace(go.Scatter(x=prod1['Date'], y=prod1['Sum_q_lamels_by_month'], mode='lines', name='PROD-suma'), row=3, col=1)
    fig.add_trace(go.Scatter(x=prod1['Date'], y=prod1['Sale_quantity_final'], mode='lines', name=f'{prod}'), row=3, col=1)
    fig.add_trace(go.Bar(x=prod1['Date'], y=prod1['Promo_ind_all'], name='Wskaźnik promocji-all'), row=4, col=1)
    fig.add_trace(go.Bar(x=prod1['Date'], y=prod1['Promo_ind'], name='Wskaźnik promocji-produkt'), row=5, col=1)

    fig.update_yaxes(title_text='Wskaźniki', row=1, col=1)
    fig.update_yaxes(title_text='Mieszkania', row=2, col=1)
    fig.update_yaxes(title_text='PRODUKT', row=3, col=1)
    fig.update_yaxes(title_text='Promo dla ALL', row=4, col=1)
    fig.update_yaxes(title_text='Promo dla Produktu', row=5, col=1)
    # fig.update_yaxes(title_text='[produkt]', row=5, col=1)
    fig.update_layout(title='Ilości sprzedanych produktów vs. wskaźniki makroekonomiczne vs. Promocje',
                    #xaxis_title='Data',
                    height=1100
                    )
    return fig


def plot_correlation_matrix(prod1):
    cormat = prod1.rename(
        columns= {
            'CPI': 'Wskaźnik Inflacji',
            'Wsk_bud_mont': 'Wskaźnik cen bud-montaż.',
            'Sale_quantity_final': 'Liczba sprzedanych sztuk [produktu]',
            'Sum_q_lamels_by_month': 'Suma wszystkich sprzedanych lameli',
            'Sum_q_other_lamels_by_month': 'Suma pozostałych lameli',
            'Promo_ind': 'Wskaźnik Promocji [produktu]',
            'Promo_ind_all': 'Wskaźnik Promocji wszyskie lamele',
            'Mieszkania' : 'Liczba mieszkań oddanych do uż.'
        })[['Wskaźnik Inflacji', 'Wskaźnik cen bud-montaż.', 'Liczba mieszkań oddanych do uż.', 
                      'Wskaźnik Promocji [produktu]', 'Wskaźnik Promocji wszyskie lamele', 
                      'Liczba sprzedanych sztuk [produktu]', 
                      'Suma wszystkich sprzedanych lameli', 
                      'Suma pozostałych lameli']].corr()
    # Mask to matrix
    mask = np.zeros_like(cormat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Viz
    df_corr_viz = cormat.mask(mask).dropna(how='all').dropna('columns', how='all')
    fig = px.imshow(round(df_corr_viz,2), aspect="auto",text_auto=True,title="Macierz korelacji")
    return fig

def main():
    st.title('Zbiór danych sprzedażowych')
    st.subheader("Próbka danych: Styczeń 2021- Czerwiec 2023")
    st.markdown("Dane dotyczą wybranych grup asortymentowych w latach 2022-2023 wygenerowały największą sprzedaż w firmie.") 
    st.markdown("Dodatkowo zbiór danych zawiera historyczne dane makroekonomiczne z tego okresu.")
    df = read_data()
    if df not in st.session_state:
        st.session_state['df'] = df
    prod = st.selectbox(
        'Wybierz produkt do analizy:',
        list(df.Prod_code.unique()))
    st.write(df)
    #st.write('Wybrany produkt:', prod)
    if prod not in st.session_state:
        st.session_state['prod'] = prod
    df_prod = data_for_product(df, prod)
    #df_prod.to_csv(f'selected_data/df_prod_{prod}.csv', index=False)
    st.markdown("### Trendy sprzedażowe dla wybranego produktu") #sumarycznie
    tab1, tab2 = st.tabs(['Wykres słupkowy', 'Wykres liniowy'])
    fig = px.bar(df_prod.rename(columns={'Date':'Data', 'Sale_quantity_final': 'Liczba sprzedanych produktów'}), x='Data', y='Liczba sprzedanych produktów',
             title=f'{prod}')
    fig_line = px.line(df_prod.rename(columns={'Date':'Data', 'Sale_quantity_final': 'Liczba sprzedanych produktów'}), x='Data', y='Liczba sprzedanych produktów',
             title=f'{prod}')
    # fig.write_json(f"plotly_figure{prod}.json")
    tab1.plotly_chart(fig, theme="streamlit", use_container_width=True)
    tab2.plotly_chart(fig_line, theme="streamlit", use_container_width=True)
    st.markdown("#### Zależności i korelacje pomiędzy zmiennymi")
    fig_data_features = plot_data_with_features(df_prod, prod)
    st.plotly_chart(fig_data_features, theme="streamlit", use_container_width=False)
    corr = plot_correlation_matrix(df_prod)
    #st.pyplot(corr.get_figure())
    st.plotly_chart(corr, theme="streamlit", use_container_width=False)
    

if __name__ == "__main__":
    main()