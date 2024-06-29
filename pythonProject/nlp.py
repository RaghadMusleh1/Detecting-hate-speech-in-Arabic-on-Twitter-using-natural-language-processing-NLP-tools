import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
dfTrain = pd.read_csv("C:\\Users\\DELL\\PycharmProjects\\HelloNLP\\pythonProject\\test.csv")
dfTest = pd.read_csv("C:\\Users\\DELL\\PycharmProjects\\HelloNLP\\pythonProject\\test.csv")

# Check for missing values
if dfTrain.isnull().values.any():
    print("Training data contains missing values. Please clean the data.")
if dfTest.isnull().values.any():
    print("Test data contains missing values. Please clean the data.")

# Define the stop words
stop_words = set(stopwords.words('arabic'))

# Define text normalization function
def normalize_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove diacritics (arabic harakats)
    text = re.sub(r'[ًٌٍَُِْ]', '', text)
    return text

# Apply tokenization, normalization, and remove stop words
def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = normalize_text(text)
    tokens = [token for token in word_tokenize(text) if token.lower() not in stop_words]
    return ' '.join(tokens)

dfTrain['Tweet'] = dfTrain['Tweet'].apply(preprocess)
dfTest['Tweet'] = dfTest['Tweet'].apply(preprocess)

# Ensure text data is not empty after preprocessing
dfTrain = dfTrain[dfTrain['Tweet'].str.strip() != '']
dfTest = dfTest[dfTest['Tweet'].str.strip() != '']

# Check if the dataset is still valid
if dfTrain.empty:
    raise ValueError("Training data is empty after preprocessing. Please provide valid data.")
if dfTest.empty:
    raise ValueError("Test data is empty after preprocessing. Please provide valid data.")

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(dfTrain['Tweet'], dfTrain['Class'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer with n-gram
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
x_train_vectorized = vectorizer.fit_transform(x_train)
x_val_vectorized = vectorizer.transform(x_val)
x_test_vectorized = vectorizer.transform(dfTest['Tweet'])

# Hyperparameters grid for SVC
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Use GridSearchCV to find the best model
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_vectorized, y_train)
best_model = grid_search.best_estimator_

# Print the best parameters
print("Best parameters: ", grid_search.best_params_)

# Evaluate the model on the validation set
y_val_pred = best_model.predict(x_val_vectorized)
print("Validation set evaluation:")
print(classification_report(y_val, y_val_pred))

# Predict the labels for the test set
y_test_pred = best_model.predict(x_test_vectorized)

# Evaluate the classifier on the test set
print("Test set evaluation:")
print(classification_report(dfTest['Class'], y_test_pred))

# Streamlit interface
st.title('Arabic Levantine Hate Speech Detection')

# Visualization: Distribution of classes in training and test datasets
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='Class', data=dfTrain, ax=ax)
ax.set_title('Training Set Class Distribution')
# Save the figure to a temporary file or a buffer
plt.savefig('train_class_distribution.png')
st.image('train_class_distribution.png', caption='Training Set Class Distribution')
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='Class', data=dfTest, ax=ax)
ax.set_title('Test Set Class Distribution')
# Save the figure to a temporary file or a buffer
plt.savefig('test_class_distribution.png')
st.image('test_class_distribution.png', caption='Test Set Class Distribution')
plt.close()

# Visualization: Confusion Matrix for the validation set
fig, ax = plt.subplots()
cm = confusion_matrix(y_val, y_val_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(ax=ax)
plt.title('Validation Set Confusion Matrix')
# Save the figure to a temporary file or a buffer
plt.savefig('validation_confusion_matrix.png')
st.image('validation_confusion_matrix.png', caption='Validation Set Confusion Matrix')
plt.close()

# Visualization: Confusion Matrix for the test set
fig, ax = plt.subplots()
cm = confusion_matrix(dfTest['Class'], y_test_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(ax=ax)
plt.title('Test Set Confusion Matrix')
# Save the figure to a temporary file or a buffer
plt.savefig('test_confusion_matrix.png')
st.image('test_confusion_matrix.png', caption='Test Set Confusion Matrix')
plt.close()

# Streamlit interface for user input
user_input = st.text_area("Enter a tweet to analyze:")

if st.button("Analyze"):
    if not user_input.strip():
        st.write("Please enter a valid tweet.")
    else:
        try:
            # Preprocess the new text
            new_text = preprocess(user_input)

            if new_text.strip() == '':
                st.write("The preprocessed tweet is empty. Please enter a valid tweet.")
            else:
                # Vectorize the new text using the same vectorizer
                new_text_vectorized = vectorizer.transform([new_text])

                # Make the prediction
                prediction = best_model.predict(new_text_vectorized)

                # Convert the prediction to the corresponding class label
                class_label = "normal" if prediction[0] == "normal" else "abusive/hate"

                st.write(f"Predicted class label: {class_label}")
        except Exception as e:
            st.write(f"An error occurred: {e}")