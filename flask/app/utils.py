import torch
import torchtext

import sys
from os.path import abspath, join, dirname

# Add the project's root directory to the Python path
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

from lstm.models.engine import LSTMModel
from lstm.utils.predict import predict_sentiment

device = torch.device('cpu')


vocab = torch.load("assets/lstm/vocab.pth", map_location=torch.device('cpu'))
pad_index = vocab["<pad>"]

model = LSTMModel(
    vocab_size=len(vocab),
    embedding_dim=600,
    hidden_dim=600,
    output_size=2,
    n_layers=2,
    bidirectional=True,
    dropout_rate=0.5,
    pad_index=pad_index
)

PATH = "assets/lstm/model_state_dict.pt"
model.load_state_dict(torch.load(PATH, map_location=device))

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def predict(text):
    return predict_sentiment(text, model, tokenizer, vocab, device="cpu")
