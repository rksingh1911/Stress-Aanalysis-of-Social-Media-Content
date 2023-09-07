import flask
import os
import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model and vectorizer from the pickle file
with open('my_model.pkl', 'rb') as model_file:
    loaded_model_data = pickle.load(model_file)

# Extract the vectorizer and model from the loaded data
tfidf_vectorizer = loaded_model_data['vectorizer']
LR = loaded_model_data['model']

# Define a function to preprocess and transform new text
def preprocess_and_transform(text):
    preprocessed_text = re.sub(r'https?://\S+|www\.\S+', '', text)
    preprocessed_text = re.sub(r'<.*?>+', '', preprocessed_text)
    preprocessed_text = preprocessed_text.lower()
    stopwords_set = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    text = [word for word in preprocessed_text.split(' ') if word not in stopwords_set]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    
    # Transform using the loaded vectorizer
    vectorized_input = tfidf_vectorizer.transform([text])
    return vectorized_input

#HTTP Address for home page
@app.route('/Rohit_App')
def home():
    return render_template('index.html')

#HTTP Address for Prediction Page
@app.route('/y_predict', methods=['POST'])
def y_predict():
    input_text = request.form['Sentence']
    print(input_text)
    
    # Preprocess and transform the new text
    V_input = preprocess_and_transform(input_text)
    
    # Make predictions using the loaded model
    prediction = LR.predict(V_input)
    print("Predicted class:", prediction)
    
    if prediction == [0]:
        output = "Not Stress"
    else:
        output = "Stress"
    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == '__main__':
    app.run()


#--------------------------------------------------------------------------------------

# Example new text for prediction
#new_text = "Sometime I feel like I need some time"

# Preprocess and transform the new text
#vectorized_input = preprocess_and_transform(new_text)

# Make predictions using the loaded model
#prediction = LR.predict(vectorized_input)

#print("Predicted class:", prediction)
