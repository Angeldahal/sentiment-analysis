import torch
from transformers import AutoModel, AutoTokenizer

import sys
from os.path import abspath, join, dirname
import torchtext

# Add the project's root directory to the Python path
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from lstm.models.engine import LSTMModel

from bert.models.engine import Transformer

def model_init(is_bert):
    if is_bert:
        model_ckpt = "bert-base-uncased"
        transformer = AutoModel.from_pretrained(model_ckpt)

        device = torch.device("cpu")

        model = Transformer(transformer, output_dim=2, freeze=False)
        model.load_state_dict(torch.load('assets/bert/model_state_dict.pt', map_location=device))

        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        return model, tokenizer

    else:   
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

        return vocab, model, tokenizer