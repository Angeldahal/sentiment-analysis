import streamlit as st
import torch
import torchtext

import sys
from os.path import abspath, join, dirname

# Add the project's root directory to the Python path
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from lstm.models.engine import LSTMModel
from lstm.utils.predict import predict_sentiment

vocab = torch.load("assets/vocab.pth", map_location=torch.device('cpu'))
model = torch.load("assets/sentiment_all_lstm.pt", map_location=torch.device('cpu'))

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

if __name__ == "__main__":
    st.title("Sentiment Analyzer!!!:bar_chart:")


    text_input = st.text_input(
        "Enter some text 👇 whose sentiment you'd like to know:",
        placeholder="I am very happy today.",
    )

    if text_input:
        probability, sentiment = predict_sentiment(text_input, model, tokenizer, vocab, device="cpu")
        sentiment = "Positive" if sentiment == 1 else "Negative"
        
        st.write(f"The model is {probability*100:.2f}% certain that the sentiment is {sentiment}")
    