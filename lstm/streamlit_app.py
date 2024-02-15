import streamlit as st
from utils.predict import predict_sentiment
import torch
import torchtext
from models.engine import LSTMModel

vocab = torch.load("assets/vocab.pth", map_location=torch.device('cpu'))
print("done")
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
        sentiment = "Positive" if sentiment > 0.5 else "Negative"
        
        st.write(f"The model is {probability*100:.2f}% certain that the sentiment is {sentiment}")
    