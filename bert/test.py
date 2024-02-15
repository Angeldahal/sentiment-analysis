import torch
import transformers

from utils.predict import predict_sentiment

model_ckpt = "bert-base-uncased"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)

text = "This is a good model."

model = torch.load("bert/assets/transformer.pt")

sentiment, probability = predict_sentiment(text, model, tokenizer, device="cpu")

print(sentiment, probability)