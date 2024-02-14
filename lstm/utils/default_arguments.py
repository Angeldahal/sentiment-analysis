from preprocess import vocab, train_data, pad_index

args = {
    "vocab_size": len(vocab),
    "embedding_dim": 600,
    "hidden_dim": 600,
    "output_dim": len(train_data.unique("target")),
    "n_layers": 2,
    "bidirectional": True,
    "dropout_rate": 0.5,
    "lr": 0.005,
    "weight_decay": 0.001,
    "pad_index": pad_index
}
