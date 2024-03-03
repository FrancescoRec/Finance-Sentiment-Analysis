import streamlit as st
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

header = st.container()
dataset = st.container()
tools = st.container()
model_training = st.container()

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

    news_headlines = scrape_yahoo_finance_news()
    if news_headlines:
        for headline in news_headlines[:3]:
            sentiment_score = analyze_sentiment(headline)
            st.write("Headline:", headline)
            st.write("  Sentiment Score:")
            st.json(sentiment_score)


    else:
        print("No news headlines scraped from Yahoo Finance")


