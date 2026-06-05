# Reti Neurali per la Regressione — PyTorch
Guida standalone e completa. Basata su `torchnn.py` + notebook di regressione.

---

## 1. Import

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from tqdm import trange
```

- **Rimuovere** `accuracy_score`, `classification_report`, `confusion_matrix`
- **Aggiungere** `r2_score`, `mean_squared_error`, `mean_absolute_error`

---

## 2. Config

```python
config = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 20,
    "patience": 5,
    "min_delta": 0.01,
    "momentum": 0.9,
    "nesterov": True,
    "train_loss": nn.MSELoss(),   # oppure nn.L1Loss() se richiesto
    "test_loss":  nn.MSELoss(),
    "metrics": [r2_score, mean_squared_error, mean_absolute_error],  # ← metriche regressione
    "metric_average": None        # ← None per regressione (nessuna media multiclasse)
}
```

---

## 3. Preparazione dei dati (step preliminare)

`stratify` non si usa nella regressione (non ci sono classi da bilanciare).

```python
X_train, X_te, y_train, y_te = train_test_split(X, y, random_state=42, test_size=0.05)
X_tr, X_val, y_tr, y_val     = train_test_split(X_train, y_train, random_state=42, test_size=0.10)
# ← niente stratify
```

---

## 4. Dataset

Il tensore y deve essere **float32**, non long.
Se i dati vengono da un DataFrame pandas, usare `.values`.

```python
class TensDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.float32)  # ← float32, NON torch.long

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

---

## 5. DataLoader

```python
train_data = TensDataset(X_tr, y_tr)
val_data   = TensDataset(X_val, y_val)
test_data  = TensDataset(X_te, y_te)

train_loader, val_loader, test_loader = make_dataloaders(
    train_data, val_data, test_data, batch=config["batch_size"]
)
```

---

## 6. Architettura della rete

- L'ultimo strato deve avere **1 solo neurone di output**
- **Nessun** `nn.LogSoftmax` alla fine
- Il metodo `forward()` deve applicare `.squeeze(1)` sull'output della rete

```python
class ReteDensa(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)      # ← 1 solo neurone di output, NESSUN LogSoftmax
        )

    def forward(self, x):
        return self.network(x).squeeze(1)  # (batch, 1) → (batch,)
```

> **Perché squeeze(1)**: l'ultimo strato produce shape `(batch_size, 1)`.
> Il target y ha shape `(batch_size,)`. `.squeeze(1)` allinea le forme
> per evitare errori nella loss e nelle metriche.

---

## 7. Device, modello, optimizer

```python
device = "cpu"  # oppure "cuda" se disponibile

model = ReteDensa(input_dim=X_tr.shape[1])
opt   = torch.optim.Adam(model.parameters(),
                         lr=config["learning_rate"],
                         betas=[0.92, 0.997])
```

---

## 8. EarlyStopping

```python
class EarlyStopping:
    def __init__(self, patience=config["patience"], min_delta=config["min_delta"]):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_validation_loss = torch.inf

    def __call__(self, validation_loss):
        if (validation_loss + self.min_delta) >= self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.min_validation_loss = validation_loss
            self.counter = 0

es = EarlyStopping()
```

---

## 9. Training loop

```python
def train_loop(model, dataloader, optimizer, device, pbar,
               loss_fn=config["train_loss"]):

    num_batches = len(dataloader)
    epoch_loss = 0.0
    model.train()

    for _, (X, y) in zip(pbar, dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        batch_loss = loss_fn(logits, y)
        epoch_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

    return epoch_loss / num_batches
```

---

## 10. Evaluation loop

- **Rimuovere** tutto ciò che riguarda `accuracy`
- **Rimuovere** `y_score` (non serve per la regressione)
- `y_pred` si raccoglie con `.cpu().detach().tolist()`, **senza** `argmax(1)`
- `epoch_metrics` viene costruito automaticamente dalla lista `metrics` in config:
  il dizionario usa `metric.__name__` come chiave → **non definirlo manualmente**

```python
def eval_loop(model, dataloader, device,
              loss_fn=config["test_loss"],
              metrics=config["metrics"],
              average=config["metric_average"]):

    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # ← niente argmax(1), niente y_score
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    test_loss /= num_batches

    # epoch_metrics costruito automaticamente dalle metriche in config
    epoch_metrics = {}
    for metric in metrics:
        epoch_metrics[metric.__name__] = metric(y_true, y_pred)

    return test_loss, epoch_metrics, y_true, y_pred
```

> Le chiavi del dizionario sono i nomi delle funzioni, ad es.:
> `epoch_metrics["r2_score"]`, `epoch_metrics["mean_squared_error"]`, `epoch_metrics["mean_absolute_error"]`

---

## 11. train_test con salvataggio del modello

Il salvataggio avviene rispetto al **miglior MSE sul validation set** (lower is better → init `float("inf")`).
Per salvare rispetto al miglior R2 (higher is better → init `-float("inf")`), sostituire il confronto con `epoch_validate_metrics["r2_score"] > best_val_r2`.

```python
def train_test(model, optimizer, device,
               train_dataloader, test_dataloader,
               epochs=config["epochs"],
               train_loss_fn=config["train_loss"],
               test_loss_fn=config["test_loss"],
               early_stopping=None,
               val_dataloader=None,
               scheduler=None,
               metrics=config["metrics"],
               average=config["metric_average"]):

    train_loss, validation_loss, test_loss = [], [], []
    # ← niente accuracy
    best_val_mse = float("inf")

    test_metrics = {}
    for metric in metrics:
        test_metrics[metric.__name__] = []

    num_batches = len(train_dataloader.batch_sampler)

    for epoch in range(1, epochs + 1):
        pbar = trange(num_batches)
        pbar.set_description(desc='Epoch {:4d}'.format(epoch))

        epoch_train_loss = train_loop(model, train_dataloader, optimizer, device, pbar,
                                      loss_fn=train_loss_fn)
        train_loss.append(epoch_train_loss)

        if val_dataloader is not None:
            epoch_validate_loss, epoch_validate_metrics, *_ = eval_loop(
                model, val_dataloader, device, loss_fn=test_loss_fn, metrics=metrics)
            validation_loss.append(epoch_validate_loss)

        epoch_test_loss, epoch_metrics, *_ = eval_loop(
            model, test_dataloader, device, loss_fn=test_loss_fn, metrics=metrics)
        test_loss.append(epoch_test_loss)

        for metric in metrics:
            test_metrics[metric.__name__].append(epoch_metrics[metric.__name__])

        val_str = f'Validation loss: {epoch_validate_loss:6.4f}\n' if val_dataloader is not None else ' '
        print(f"Train loss: {epoch_train_loss:6.4f}\n{val_str}Test loss: {epoch_test_loss:6.4f}")
        for label, metric in epoch_metrics.items():
            print(f'{label}: {metric:6.2f}', end=' ')
        print()

        if early_stopping is not None:
            # salvataggio rispetto al miglior MSE sul validation
            if epoch_validate_metrics["mean_squared_error"] < best_val_mse:
                best_val_mse = epoch_validate_metrics["mean_squared_error"]
                save_model(model, optimizer, epoch,
                           train_loss, validation_loss, test_loss, test_metrics,
                           path=f"best_mod_{epoch}.pth")
            early_stopping(epoch_validate_loss)
            if early_stopping.early_stop:
                break

        if scheduler is not None:
            scheduler.step()

    return train_loss, validation_loss, test_loss, test_metrics
```

---

## 12. save_model

```python
def save_model(net, optimizer, current_epoch,
               train_loss, val_loss, test_loss, metrics, path):

    to_save = {
        'epoch': current_epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': train_loss,
        'validation_loss': val_loss,
        'test_loss': test_loss
        # ← niente accuracy
    }
    to_save.update(metrics)
    torch.save(to_save, path)
```

---

## 13. load_model

Il modello e l'ottimizzatore devono essere **già inizializzati** prima di chiamare la funzione.

```python
def load_model(path, model, optimizer, device=None):
    if device is not None and device != 'cpu':
        model.to(device)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint
```

Utilizzo:
```python
model, optimizer, checkpoint = load_model("best_mod_15.pth", model, opt, device)
```

---

## 14. displayMetrics e displayLosses

```python
def displayLosses(train_loss, test_loss, validation_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='training loss')
    plt.plot(epochs, validation_loss, label='validation loss')
    plt.plot(epochs, test_loss, label='test loss')
    plt.legend(loc='lower right')
    plt.title("Loss di addestramento")
    plt.show()

def displayMetrics(metrics):
    # ← niente accuracy, niente riferimenti a classi
    epochs = range(1, len(next(iter(metrics.values()))) + 1)
    for label, metric in metrics.items():
        plt.plot(epochs, metric, label=label)
    plt.legend(loc='lower right')
    plt.title("Metriche")
    plt.show()
```

---

## 15. Valutazione finale sul test set

```python
test_loss_val, test_epoch_metrics, y_true, y_pred = eval_loop(model, test_loader, device)
print(f"R2  : {test_epoch_metrics['r2_score']:.4f}")
print(f"MSE : {test_epoch_metrics['mean_squared_error']:.4f}")
print(f"MAE : {test_epoch_metrics['mean_absolute_error']:.4f}")
```

---

## 16. Checklist riassuntiva

| Cosa | Classificazione | Regressione |
|------|----------------|-------------|
| Loss | `nn.CrossEntropyLoss()` | `nn.MSELoss()` / `nn.L1Loss()` |
| dtype y | `torch.long` | `torch.float32` |
| Ultimo strato | N neuroni + `LogSoftmax` | 1 neurone, niente softmax |
| `forward()` | — | `.squeeze(1)` |
| `y_pred` | `argmax(1)` | `.cpu().detach().tolist()`, niente argmax |
| `y_score` | sì (per AUC/ROC) | non serve |
| `epoch_metrics` | costruito da metrics list | costruito da metrics list (stesso meccanismo) |
| Metriche in config | `accuracy_score`, F1, AUC | `r2_score`, `mean_squared_error`, `mean_absolute_error` |
| `metric_average` | `"macro"` o `"weighted"` | `None` |
| `stratify` in split | sì | no |
| Init best per salvataggio | accuracy: `0.0` o `-inf` | MSE: `float("inf")` / R2: `-float("inf")` |
| Confronto per salvataggio | accuracy: `>` | MSE: `<` / R2: `>` |
| `accuracy` in save_model | sì | no (rimossa) |