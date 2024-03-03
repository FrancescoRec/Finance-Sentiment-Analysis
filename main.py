import streamlit as st
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import joblib
import requests
from io import BytesIO
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd



header = st.container()
dataset = st.container()
tools = st.container()
model_training = st.container()
test = st.container()

import nltk
nltk.download('wordnet')

def convert_to_lower(text):
    if text is None:
        return ""
    return text.lower()

def remove_punct(text):
    if text is None:
        return ""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenise(text):
    if text is None:
        return ""
    return word_tokenize(text)

def remove_stop_words(tokens):
    if tokens is None:
        return ""
    return [word for word in tokens if word.lower() not in set(stopwords.words('english'))]

def remove_empty_strings(tokens):
    if tokens is None:
        return ""
    return [token for token in tokens if token.strip()]

def lemmatise(tokens):
    if tokens is None:
        return ""
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]


def cleaning_text(text):
    if text is None:
        return []

    text = convert_to_lower(text)
    text = remove_punct(text)
    token = tokenise(text)
    token = remove_stop_words(token)
    token = remove_empty_strings(token)
    token = lemmatise(token)
    return token

# Function to scrape news articles from Yahoo Finance
def scrape_yahoo_finance_news():
    url = "https://finance.yahoo.com/news/"

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = soup.find_all('h3')
        news_headlines = [headline.get_text() for headline in headlines]
        return news_headlines
    else:
        print(response.content)
        print("Failed to fetch data from Yahoo Finance")

# Function to analyze sentiment using VADER
def analyze_sentiment(headline):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(headline)
    return sentiment_score


with header:
    st.title("Welcome to our Finance Sentiment Analysis project")
    st.text("Matilde Bernocchi, Francesco D'aleo, and Elena Ginebra")
    

with dataset:
    st.header("Dataset")
    st.subheader("Yahoo Finance")
    st.markdown("The website Yahoo Finance is a platform providing financial news and updates, offering articles covering various topics such as market trends, investment advice, and economic analysis.")
    st.markdown("**Why did we choose Yahoo Finance?**  has a comprehensive coverage of financial news, updates, and analysis, offering insights into market trends, investment opportunities, and economic developments, making it a valuable resource for staying informed about the financial world.")
    


with model_training:
    st.header("Training Model")
    st.markdown("After collecting the headlines from Yahoo Finance, we have to apply sentmient analysis to the dataset in order to retrieve a score that ranks the headline on a score that is: postive, negative or neutral")
    st.subheader("vaderSentiment")
    st.markdown("VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.")


    
    list_of_headlines = []
    news_headlines = scrape_yahoo_finance_news()
    
    if news_headlines:
        for headline in news_headlines[6:11]:
            sentiment_score = analyze_sentiment(headline)
            #st.write("Headline:", headline)
            #st.write("  Sentiment Score:")
            #st.json(sentiment_score)
            list_of_headlines.append(headline)
        
        financial_phrasebank = load_dataset("financial_phrasebank", 'sentences_50agree')
        train_data_finance = financial_phrasebank["train"]
        train_data_finance = pd.DataFrame(train_data_finance)
        train_data_finance['proc_sentence'] = train_data_finance['sentence'].apply(cleaning_text)
        
        X = train_data_finance['proc_sentence']
        y = train_data_finance['label']
        X_list = [' '.join(words) for words in X]
        X_train, X_test, y_train, y_test = train_test_split(X_list, y, test_size=0.2, random_state=42)         


        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        print("Doing random forest...")
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        print("DONE")
        X_test_tfidf = tfidf_vectorizer.transform(list_of_headlines)
        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        random_forest_model.fit(X_train_tfidf, y_train)
        y_pred_rf = random_forest_model.predict(X_test_tfidf)
        st.write(y_pred_rf)
        st.write("SIZES", len(y_pred_rf), len(list_of_headlines))
        
        prediction_texts = {0: "Negative", 1: "Neutral", 2: "Positive"}
        textual_predictions = [prediction_texts[pred] for pred in y_pred_rf]
        
        results_df = pd.DataFrame({
            'Headline': list_of_headlines,
            'Prediction': textual_predictions
        })
        st.write(results_df)






with test:
    st.header("TEST OUR SENTIMENT ANALYSER!")
    if st.button('Check todays headlines!'):
        news_headlines = scrape_yahoo_finance_news()
        if news_headlines:
            for headline in news_headlines[:20]:
                sentiment_score = analyze_sentiment(headline)
                st.write("Headline:", headline)
                st.write("  Sentiment Score:")
                st.json(sentiment_score)
    else:
        st.write('Press the button ')
