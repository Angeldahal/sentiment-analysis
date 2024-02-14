import streamlit as st
from utils.predict import predict_output


if __name__ == "__main__":
    st.title("Sentiment Analyzer!!!:bar_chart:")


    text_input = st.text_input(
        "Enter some text 👇 whose sentiment you'd like to know:",
        placeholder="I am very happy today.",
    )

    if text_input:
        probability, sentiment = predict_output(text_input)
        sentiment = "Positive" if sentiment == 1 else "Negative"
        
        st.write(f"The model is {probability}% certain that the sentiment is {sentiment}")
    