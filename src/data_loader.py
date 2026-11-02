import pandas as pd
import streamlit as st

FAIXAS_ETARIAS = ["<45", "45-54", "55-64", "65+"]


@st.cache_data
def load_data(path="data/clean/dataset.csv"):
    """Lê o dataset já tratado (exportado ao final do notebook eda_posterior_rafaelPadilha)."""
    df = pd.read_csv(path)
    df["faixa_etaria"] = pd.Categorical(df["faixa_etaria"], categories=FAIXAS_ETARIAS, ordered=True)
    return df
