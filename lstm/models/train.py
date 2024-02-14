import numpy as np
from tqdm.auto import tqdm
import torch


def get_accuracy(self, prediction, label):

    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size

    return accuracy


def train(self,
          dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module, 
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          device: str = "cpu"
):
    model.train()
    epoch_losses = []
    epoch_accs = []

    for batch in tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = loss_fn(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

    return np.mean(epoch_losses), np.mean(epoch_accs)

    
def evaluate(self, 
             dataloader: torch.utils.data.DataLoader, 
             model: torch.nn.Module, 
             loss_fn: torch.nn.Module, 
             device: str = 'cpu'
):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = loss_fn(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)