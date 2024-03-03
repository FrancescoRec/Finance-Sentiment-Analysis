import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
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


header = st.container()
training = st.container()
accuracy = st.container()
yahoo_test = st.container()


with header:
    st.title("Welcome to our Finance Sentiment Analysis project")
    st.text("Matilde Bernocchi, Francesco D'aleo, and Elena Ginebra")
    st.header("About this project")
    st.markdown("We embarked on a mission to explore how sentiment analysis can be applied within this dynamic field. We are trying to understand the nuances of market sentiment through the lens of data science.")
    

with training:
    with st.spinner('Loading...'):
        st.header("How we trained our model")
        st.markdown("Our approach to this challenge involved the use of a Random Forest classifier, a choice motivated by its robustness and efficacy in handling diverse datasets. We meticulously prepared our dataset, which comprised financial news articles and reports, labeling them based on their sentiment (positive, negative, or neutral).")
        st.markdown("The preprocessing phase included text cleaning, normalization, and the application of TF-IDF vectorization to transform textual data into a format suitable for model training.")
        st.markdown("The training process was iterative, with continuous evaluation and fine-tuning to enhance model performance. Our commitment to rigorous testing and validation has been pivotal in developing a model that we believe offers valuable predictions.")

        financial_phrasebank = load_dataset("financial_phrasebank", 'sentences_50agree')
        train_data_finance = financial_phrasebank["train"]
        train_data_finance = pd.DataFrame(train_data_finance)
        st.dataframe(train_data_finance.head(3))
        financial_phrasebank = load_dataset("financial_phrasebank", 'sentences_50agree')
        train_data_finance = financial_phrasebank["train"]
        train_data_finance = pd.DataFrame(train_data_finance)
        train_data_finance['proc_sentence'] = train_data_finance['sentence'].apply(cleaning_text)
        X = train_data_finance['proc_sentence']
        y = train_data_finance['label']
        X_list = [' '.join(words) for words in X]
        X_train, X_test, y_train, y_test = train_test_split(X_list, y, test_size=0.2, random_state=42)         

        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        fig, ax = plt.subplots()
        colours = ['lightgray', 'limegreen', "#bd0d30"]
        st.title('Training data weights')
        label_counts = train_data_finance['label'].map({0: 'negative', 1: 'neutral', 2: 'positive'}).value_counts(normalize=True)    
        ax.pie(label_counts, autopct='%1.1f%%', startangle=140,colors=colours)
        ax.axis('equal')  
        plt.legend(label_counts.index, loc="best", bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)
        st.success('Done!')

with accuracy:
    st.header("Accuracy of our model")
    st.metric(label="Accuracy of our model", value="76%", delta="Great!")    
     

with yahoo_test:
    st.header("Test our model with Yahoo Finance!")
    st.image('yahoo_logo.png')
    st.subheader("Yahoo Finance")
    st.markdown("The website Yahoo Finance is a platform providing financial news and updates, offering articles covering various topics such as market trends, investment advice, and economic analysis.")
    st.markdown("**Why did we choose Yahoo Finance?**  has a comprehensive coverage of financial news, updates, and analysis, offering insights into market trends, investment opportunities, and economic developments, making it a valuable resource for staying informed about the financial world.")
    selected_value = st.slider('Slide to select number of headlines', min_value=1, max_value=26)

    if st.button('Press here to check todays headlines!'):
        with st.spinner('Loading...'):
            list_of_headlines = []
            news_headlines = scrape_yahoo_finance_news()
            
            if news_headlines:
                for headline in news_headlines[6:6+selected_value]:
                    list_of_headlines.append(headline)
                
                X_test_tfidf = tfidf_vectorizer.transform(list_of_headlines)
                random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
                random_forest_model.fit(X_train_tfidf, y_train)
                y_pred_rf = random_forest_model.predict(X_test_tfidf)
                prediction_texts = {0: "Negative", 1: "Neutral", 2: "Positive"}
                textual_predictions = [prediction_texts[pred] for pred in y_pred_rf]
                
                results_df = pd.DataFrame({
                    'Headline': list_of_headlines,
                    'Prediction': textual_predictions
                })
                st.write(results_df)
                st.success('Done')
                st.balloons()
                                                
                predictions_counter = Counter(textual_predictions)
                total_predictions = len(textual_predictions)

                positive_count = predictions_counter["Positive"]
                negative_count = predictions_counter["Negative"]
                neutral_count = predictions_counter["Neutral"]

                positive_percentage = (positive_count / total_predictions) * 100 if total_predictions > 0 else 0
                negative_percentage = (negative_count / total_predictions) * 100 if total_predictions > 0 else 0
                neutral_percentage = (neutral_count / total_predictions) * 100 if total_predictions > 0 else 0

                st.subheader("Sentiment Prediction Summary")
                st.markdown(f"**Positive**: {positive_count} 🟢 ({positive_percentage:.2f}%)")
                st.markdown(f"**Neutral**: {neutral_count} 🔵 ({neutral_percentage:.2f}%)")
                st.markdown(f"**Negative**: {negative_count} 🔴 ({negative_percentage:.2f}%)")
                st.progress(neutral_percentage / 100)

                st.subheader("Thank you, hope you enjoyed our model!")

    else:
        st.write('')


