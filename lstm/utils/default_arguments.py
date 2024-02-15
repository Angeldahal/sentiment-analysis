# from preprocess import vocab, train_data, pad_index
import torch
vocab = torch.load("lstm/assets/vocab.pth", map_location=torch.device('cpu'))
pad_index = vocab["<pad>"]

args = {
    "vocab_size": len(vocab),
    "embedding_dim": 600,
    "hidden_dim": 600,
    "output_size": 2,
    "n_layers": 2,
    "bidirectional": True,
    "dropout_rate": 0.5,
    "lr": 0.005,
    "weight_decay": 0.001,
    "pad_index": pad_index
}

print(args["embedding_dim"])