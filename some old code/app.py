import random
from nltk.stem import PorterStemmer
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

def check(query):
    filtered_text = preprocess_text(query)
    
    with open('./Assets/tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)
    tokenized_text = tokenizer.texts_to_sequences([filtered_text])
    max_sequence_length = 50

    padded_sequences = pad_sequences(tokenized_text, maxlen=max_sequence_length, padding='post')
    
    model = keras.models.load_model('./Assets/model.h5')
    
    probability = model.predict(padded_sequences)[0]
    
    if probability < 0.5:
        return 'negative'
    return 'positive'

def preprocess_text(text):
    #Remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove non-alphanumeric characters and special characters
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    
    # Remove user mentions and hashtags
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"#[^\s]+", "", text)
    
    # Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # processed_text = applying_stemming(filtered_tokens)
    
    return filtered_tokens


import streamlit as st

st.title("Sentiment Analysis")

user_input = st.text_input(
    'Enter the text whose Sentiment you want to analyze',

)

if 'last_text_input' not in st.session_state:
    st.session_state.last_text_input = ''

if st.button('Analyze') or (user_input and st.session_state.last_text_input != user_input):
    result = check(user_input)
    st.write("The model predict that the statement is: "+result)

