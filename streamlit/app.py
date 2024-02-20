import streamlit as st

from model_init import model_init

import sys
from os.path import abspath, join, dirname

# Add the project's root directory to the Python path
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from lstm.utils.predict import predict_sentiment
from bert.utils.predict import predict_sentiment_bert

if __name__ == "__main__":
    st.title("Sentiment Analyzer!!!:bar_chart:")


    text_input = st.text_input(
        "Enter some text 👇 whose sentiment you'd like to know:",
        placeholder="I am very happy today.",
    )

    model = st.radio("Which model would you like to use?", ["BERT", "LSTM"])

    if text_input and model == "LSTM":
        
        vocab, model, tokenizer = model_init(is_bert=False)
        probability, sentiment = predict_sentiment(text_input, model, tokenizer, vocab, device="cpu")
        sentiment = "Positive" if sentiment == 1 else "Negative"
        
        st.write(f"The model is {probability*100:.2f}% certain that the sentiment is {sentiment}")
    
    if text_input and model == "BERT":
        model, tokenizer = model_init(is_bert=True)

        sentiment, probability = predict_sentiment_bert(text_input, model, tokenizer, device="cpu")
        sentiment = "Positive" if sentiment == 1 else "Negative"

        st.write(f"The model is {probability*100:.2f}% certain that the sentiment is {sentiment}")
