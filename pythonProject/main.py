import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

import torch

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import addWord

nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()

sentiment_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02")

from addWord import add_custom_hate_speech_words

print('test')


def remove_diacritics(word):
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED\u08D4-\u08E2]')
    return re.sub(arabic_diacritics, '', word)


def standardize_characters(word):
    word = word.replace('أ', 'ا')
    word = word.replace('إ', 'ا')
    word = word.replace('آ', 'ا')
    word = word.replace('ى', 'ي')
    word = word.replace('ة', 'ه')
    word = word.replace('ؤ', 'و')
    word = word.replace('ئ', 'ي')
    return word


def remove_punctuation(word):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟'
    return ''.join(char for char in word if char not in punctuation)


def replace_ligatures(word):
    word = word.replace('ﻻ', 'لا')
    word = word.replace('ﻷ', 'لأ')
    word = word.replace('ﻹ', 'لإ')
    word = word.replace('ﻵ', 'لآ')
    return word


normalized_stopwords = []

for word in arabic_stopwords:
    word = remove_diacritics(word)
    word = standardize_characters(word)
    word = remove_punctuation(word)
    word = replace_ligatures(word)
    normalized_stopwords.append(word)
arabic_stopwords = normalized_stopwords


def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    words = tweet.split()
    stemmed_words = [stemmer.stem(word) for word in words if word not in arabic_stopwords]
    tweet = ' '.join(stemmed_words)

    return tweet


def analyze_sentiment(tweet):
    inputs = sentiment_tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = sentiment_model(**inputs)
    scores = outputs[0][0].detach().numpy()
    probabilities = np.exp(scores) / np.sum(np.exp(scores))

    if probabilities[0] > 0.6:
        return 'Hate'
    elif probabilities[1] > 0.6:
        return 'Abusive'
    else:
        return 'Not Hate'


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Cleaned_Tweet'] = df['Tweet'].apply(preprocess_tweet)
    return df


def train_and_evaluate(train_df, test_df):
    X_train = train_df['Cleaned_Tweet']
    y_train = train_df['Class']
    X_test = test_df['Cleaned_Tweet']
    y_test = test_df['Class']

    global hate_speech_model, vectorizer, train_data
    vectorizer = TfidfVectorizer(max_features=5000)
    hate_speech_model = Pipeline([
        ('vect', vectorizer),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    hate_speech_model.fit(X_train, y_train)
    y_pred = hate_speech_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    train_data = train_df

    result_text = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}"
    return result_text


def run_model(a, b):
    st.header('Accuracy And Classification Report')
    result = train_and_evaluate(a, b)
    st.write([result])


def show_sample_tweets():
    sample_tweets = train_data.sample(n=10)
    tmp = addWord.custom_hate_speech_words
    cnt = 1
    for index, row in sample_tweets.iterrows():
        tweet = row['Tweet']
        cleaned_tweet = row['Cleaned_Tweet']
        tweet_features = vectorizer.transform([cleaned_tweet])
        feature_names = vectorizer.get_feature_names_out()
        hate_words = [feature_names[i] for i in tweet_features.nonzero()[1] if
                      hate_speech_model.named_steps['clf'].coef_[0, i] > 0]
        hate_words += [word for word in cleaned_tweet.split() if word in tmp]
        hate_words_count = len(hate_words)
        classification = 'Hate Speech' if hate_words_count > 0 else 'Not Hate Speech'
        sentiment = analyze_sentiment(tweet)
        st.write(f"TweetNumber {cnt}")
        cnt += 1
        table_data = [
            ["Tweet", tweet],
            ["Class", classification],
            ["Hate Words Count", hate_words_count],
            ["Hate Words", ', '.join(hate_words) if hate_words_count > 0 else "None"]
        ]

        table_df = pd.DataFrame(table_data, columns=["Attribute", "Value"])

        st.table(table_df)


def check_tweet(tweet):
    cleaned_tweet = preprocess_tweet(tweet)
    tweet_features = vectorizer.transform([cleaned_tweet])
    feature_names = vectorizer.get_feature_names_out()
    hate_words = [feature_names[i] for i in tweet_features.nonzero()[1] if
                  hate_speech_model.named_steps['clf'].coef_[0, i] > 0]
    hate_words += [word for word in cleaned_tweet.split() if word in addWord.custom_hate_speech_words]
    hate_words_count = len(hate_words)
    classification = 'Hate Speech' if hate_words_count > 0 else 'Not Hate Speech'
    sentiment = analyze_sentiment(tweet)
    table_data = [
        ["Tweet", tweet],
        ["Class", classification],
        ["Hate Words Count", hate_words_count],
        ["Hate Words", ', '.join(hate_words) if hate_words_count > 0 else "None"]
    ]

    table_df = pd.DataFrame(table_data, columns=["Attribute", "Value"])

    st.table(table_df)


data = []


def populate_table():
    global data
    data = addWord.custom_hate_speech_words


def gui(train_df, test_df):
    for i in range(5):
        st.write("")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Class', data=train_df, ax=ax)
    ax.set_title('Training Set Class Distribution')
    # Save the figure to a temporary file or a buffer
    plt.savefig('train_class_distribution.png')
    st.image('train_class_distribution.png', caption='Training Set Class Distribution')
    plt.close()

    for i in range(5):
        st.write("")

    fi, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Class', data=test_df, ax=ax)
    ax.set_title('Test Set Class Distribution')
    # Save the figure to a temporary file or a buffer
    plt.savefig('test_class_distribution.png')
    st.image('test_class_distribution.png', caption='Test Set Class Distribution')
    plt.close()
    for i in range(7):
        st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        user_input = st.text_area("Add Custom Hate Word:" )
        if st.button("Add"):
            if not user_input.strip():
                st.warning("Please enter a valid word.")
            else:
                try:
                    newWord = preprocess_tweet(user_input)

                    if newWord.strip() == '':
                        st.write("The preprocessed Word is empty. Please enter a valid Word.")
                    else:
                        addWord.add_custom_hate_speech_words(newWord)
                        custom_hate_speech = addWord.custom_hate_speech_words
                        print(len(custom_hate_speech))
                        st.success(f"Added '{newWord}' successfully.")


                except Exception as e:
                    st.write(f"An error occurred: {e}")

    with col2:
        st.write("Custom Hate Words")
        table = st.table(data)

        if st.button("Show custom Hate Words"):

            try:
                populate_table()
                table.table(data)

            except Exception as e:
                st.write(f"An error occurred: {e}")

    with col3:
        user_input2 = st.text_area("Add Custom Hate Word to delete")
        if st.button("Add Word"):
            if not user_input2.strip():
                st.warning("Please enter a valid word.")
            else:
                try:
                    if user_input2.strip() == '':
                        st.write("The Word is empty. Please enter a valid Word.")
                    else:
                        b = addWord.deleteWord(user_input2)
                        if b == 1:
                            st.write(f"The word {user_input2} is deleted successfully")
                        else:
                            st.write(f"The word {user_input2} is not found in the list")


                except Exception as e:
                    st.write(f"An error occurred: {e}")

    for i in range(7):
        st.write("")

    tweetInput = st.text_area("Add a Tweet:")

    if st.button("Check Tweet"):
        newTweet = ""
        if not tweetInput.strip():
            st.warning("Please enter a valid word.")
        else:
            try:
                newTweet = preprocess_tweet(tweetInput)

                if newTweet.strip() == '':
                    st.write("Normal")
                else:
                    check_tweet(newTweet)


            except Exception as e:
                st.write(f"An error occurred: {e}")

    for i in range(7):
        st.write("")

    if st.button("show sample tweets"):
        show_sample_tweets()


def select_train_file():
    st.title('Arabic Hate Speech Detection')
    for i in range(5):
        st.write("")
    train_df = None
    test_df = None
    uploaded_file = st.file_uploader("Upload the train file", key="train_file", type=["csv"])
    uploaded_file2 = st.file_uploader("Upload the test file", key="test_file", type=["csv"])

    if uploaded_file is not None:
        train = pd.read_csv(uploaded_file)
        train['Cleaned_Tweet'] = train['Tweet'].apply(preprocess_tweet)
        train_df = train
        train_file_label = st.empty()
        train_file_label.text(f"Train File: {uploaded_file.name}")
    if uploaded_file2 is not None:
        test = pd.read_csv(uploaded_file2)
        test['Cleaned_Tweet'] = test['Tweet'].apply(preprocess_tweet)
        test_df = test
        test_file_label = st.empty()
        test_file_label.text(f"Test File: {uploaded_file2.name}")

    if uploaded_file != None and uploaded_file2 != None:
        if __name__ == '__main__':
            run_model(train_df, test_df)
            gui(train_df, test_df)


select_train_file()


