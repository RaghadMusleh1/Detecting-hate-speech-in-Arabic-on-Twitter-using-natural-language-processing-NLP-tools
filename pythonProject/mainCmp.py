import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import filedialog, Text
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import addWord

nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

custom_hate_speech_words = []
loaded_tweets = pd.DataFrame()
abusive_words_list = []
hate_speech_words_list = []
data = []

def preprocess_tweet(twet):
    twet = re.sub(r'http\S+|www\S+', '', twet, flags=re.MULTILINE)
    twet = re.sub(r'\@\w+|\#', '', twet)
    twet = re.sub(r'[^\w\s]', '', twet)
    twet = ' '.join([word for word in twet.split() if word not in arabic_stopwords])
    return twet


def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    words = []
    for line in lines:
        parts = line.split(',')
        if len(parts) > 1:
            if parts[1].strip().startswith('1'):
                words.append(parts[0])
        elif len(parts) == 1:  # No rating, add word with default rating
            words.append(parts[0])
    return words


def load_words_model(abusive_words_file_path, hate_speech_words_file_path):
    global abusive_words_list, hate_speech_words_list
    if abusive_words_file_path and hate_speech_words_file_path:
        abusive_words_list = load_words(abusive_words_file_path)
        hate_speech_words_list = load_words(hate_speech_words_file_path)
        st.write("Abusive and Hate Speech words loaded successfully.")
    else:
        st.write("Please select both abusive and hate speech words files.")


def classify_tweet(tweet):
    cleaned_tweet = preprocess_tweet(tweet)
    words = cleaned_tweet.split()

    hate_words = set(word for word in words if word in hate_speech_words_list or word in custom_hate_speech_words)
    abusive_words = set(word for word in words if word in abusive_words_list or word in custom_hate_speech_words)

    if hate_words:
        classification = 'Hate Speech'
    elif abusive_words:
        classification = 'Abusive'
    else:
        classification = 'Normal'

    return classification, len(hate_words), len(abusive_words), hate_words, abusive_words


def show_sample_tweets():
    global table_data
    if not loaded_tweets.empty:
        print("11")
        sample_tweets = loaded_tweets.sample(n=10)
        cnt = 1
        for index, row in sample_tweets.iterrows():
            tweet = row['Tweet']
            classification, hate_words_count, abusive_words_count, hate_words, abusive_words = classify_tweet(tweet)
            st.write(f"TweetNumber {cnt}")
            cnt += 1
            table_data = [
                ["Tweet", tweet],
                ["Class", classification],
                ["Hate Words Count", hate_words_count],
                ["Hate Words", ', '.join(hate_words) if hate_words_count > 0 else "None"],
                ["Abusive Words Count", abusive_words_count],
                ["Abusive Words", ', '.join(abusive_words) if abusive_words_count > 0 else "None"]
            ]

            table_df = pd.DataFrame(table_data, columns=["Attribute", "Value"])

            st.table(table_df)


def check_tweet(tweet):
    cleaned_tweet = preprocess_tweet(tweet)
    classification, hate_words_count, abusive_words_count, hate_words, abusive_words = classify_tweet(cleaned_tweet)
    table_data = [
        ["Tweet", tweet],
        ["Class", classification],
        ["Hate Words Count", hate_words_count],
        ["Hate Words", ', '.join(hate_words) if hate_words_count > 0 else "None"],
        ["Abusive Words Count", abusive_words_count],
        ["Abusive Words", ', '.join(abusive_words) if abusive_words_count > 0 else "None"]
    ]
    table_df = pd.DataFrame(table_data, columns=["Attribute", "Value"])

    st.table(table_df)


def run_evaluation():
    if not loaded_tweets.empty:
        tweets = loaded_tweets['Tweet']
        y_true = loaded_tweets['Class']
        y_pred = [classify_tweet(tweet)[0] for tweet in tweets]

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)

        result = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}"
        print (10)


def populate_table():
    global data
    data = addWord.custom_hate_speech_words


def gui(train_df):
    run_evaluation()
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

    col1, col2, col3 = st.columns(3)
    with col1:
        user_input = st.text_area("Add Custom Hate Word:")
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

    uploaded_file = st.file_uploader("Upload abusive words file", key="abusive", type=["txt"])
    uploaded_file2 = st.file_uploader("Upload hate words file", key="hate", type=["txt"])
    uploaded_file3 = st.file_uploader("Upload tweet file", key="tweet", type=["csv"])

    # Placeholder for file name display
    train_file_label = st.empty()
    hate_file_label = st.empty()
    tweet_file_label = st.empty()
    global tweet, loaded_tweets

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        train_file_label.text(f"Abusive Words File: {uploaded_file.name}")
        st.write("Abusive Words File Saved at:", temp_file_path)

    if uploaded_file2 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_file.write(uploaded_file2.read())
            temp_file_path2 = temp_file.name
        hate_file_label.text(f"Hate Words File: {uploaded_file2.name}")
        st.write("Hate Words File Saved at:", temp_file_path2)

    if uploaded_file3 is not None:
        tweet = pd.read_csv(uploaded_file3)
        loaded_tweets = tweet
        tweet['Cleaned_Tweet'] = tweet['Tweet'].apply(preprocess_tweet)
        tweet_file_label.text(f"Tweet File: {uploaded_file3.name}")
        st.write("Tweet Data:", tweet.head())

    if uploaded_file is not None and uploaded_file2 is not None:
        load_words_model(temp_file_path, temp_file_path2)
        gui(tweet)


select_train_file()
