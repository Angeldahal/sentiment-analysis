import collections
from models.train import train, evaluate
import torch
# from preprocess import train_data_loader, valid_data_loader
from models.engine import LSTMModel
from utils.default_arguments import args

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = args.lr
weight_decay = args.weight_decay

model = LSTMModel(
    vocab_size=args.vocab_size,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    output_size=args.output_size,
    bidirectional=args.bidirectional,
    dropout_rate=args.dropout_rate,
    pad_index=args.pad_index,
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

n_epochs = 5
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, loss_fn, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, loss_fn, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "../model/lstm.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")