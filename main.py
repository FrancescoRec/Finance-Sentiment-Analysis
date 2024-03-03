import datasets
import streamlit as st

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Welcome to our Finance Sentiment Analysis project")
    st.text("Matilde Bernocchi, Francesco D'aleo, and Elena Ginebra")

with dataset:
    st.header("Dataset we are using")
    st.text("Reviews of starbucks (francesco)")
    financial_phrasebank = datasets.load_dataset("financial_phrasebank", "sentences_50agree")
    st.write(financial_phrasebank)


with features:
    st.header("features we are using")
    st.text("we are using features blablbala...")

with model_training:
    st.header("Model we are using")
    st.text("we are using model blablbala...")