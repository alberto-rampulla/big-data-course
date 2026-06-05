# DNN o CNN per Classificazione (Binaria e Multiclasse)

> Segui i passi **nell'ordine esatto** riportato qui sotto.

---

## PASSO 1 — Split del dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_te, y_train, y_te = train_test_split(X, y, test_size=0.10, random_state=42)
X_tr, X_val, y_tr, y_val     = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
```

Applicare `StandardScaler` se richiesto:

```python
from sklearn.preprocessing import StandardScaler
sc   = StandardScaler()
X_tr  = sc.fit_transform(X_tr)
X_val = sc.transform(X_val)
X_te  = sc.transform(X_te)
```

---

## PASSO 2 — Import da `torchnn.py` + device

Copiare questo blocco esattamente com'è:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import trange
from inspect import signature

device = "cpu"
```

---

## PASSO 3 — `config`

Copiare il config e impostare **subito** `NLLLoss` su entrambe le loss.
Le metriche e `metric_average` non vanno cambiate per la classificazione:

```python
config = {
    "learning_rate": 1e-3,       # ← modifica in base alla richiesta
    "batch_size": 64,            # ← modifica in base alla richiesta
    "epochs": 20,                # ← modifica in base alla richiesta
    "patience": 5,               # ← modifica in base alla richiesta
    "min_delta": 0.01,           # ← modifica in base alla richiesta
    "momentum": 0.9,
    "nesterov": True,
    "train_loss": nn.NLLLoss(),  # ← sempre NLLLoss per classificazione
    "test_loss":  nn.NLLLoss(),  # ← sempre NLLLoss per classificazione
    "metrics": [precision_score, recall_score, f1_score],  # non cambiare
    "metric_average": 'macro'                              # non cambiare
}
```

> ⚠️ Il `torchnn.py` originale ha `CrossEntropyLoss` come `train_loss` — va sempre
> sostituita con `NLLLoss` se usi `LogSoftmax` nell'ultimo layer.

---

## PASSO 4 — `reset_index` su tutti i dataset

Necessario per evitare disallineamenti di indice nei tensori:

```python
X_tr  = pd.DataFrame(X_tr).reset_index(drop=True)
X_val = pd.DataFrame(X_val).reset_index(drop=True)
X_te  = pd.DataFrame(X_te).reset_index(drop=True)
y_tr  = pd.Series(y_tr).reset_index(drop=True)
y_val = pd.Series(y_val).reset_index(drop=True)
y_te  = pd.Series(y_te).reset_index(drop=True)
```

> Se `X_tr` è già un DataFrame dopo lo scaler, ometti `pd.DataFrame(...)` e chiama
> direttamente `.reset_index(drop=True)`.

---

## PASSO 5 — Classe `TensDataset`

`torch.long` va bene sia per binario che per multiclasse:

```python
class TensDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.long)   # long per classificazione
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

---

## PASSO 6 — Creazione dei tensori

```python
train_data = TensDataset(X_tr,  y_tr)
val_data   = TensDataset(X_val, y_val)
test_data  = TensDataset(X_te,  y_te)
```

---

## PASSO 7 — `make_dataloaders`

Copiare la funzione da `torchnn.py` e modificare i tre parametri indicati:

```python
def make_dataloaders(train_data, val_data, test_data,
                     batch=config["batch_size"], prefetch=None, no_pagemem=False):
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch,
                                  shuffle=True,
                                  num_workers=0,           # ← 0
                                  prefetch_factor=prefetch, # ← None
                                  pin_memory=no_pagemem)   # ← False
    val_dataloader   = DataLoader(val_data,
                                  batch_size=batch,
                                  shuffle=True,
                                  num_workers=0,
                                  prefetch_factor=prefetch,
                                  pin_memory=no_pagemem)
    test_dataloader  = DataLoader(test_data,
                                  batch_size=batch,
                                  shuffle=True,
                                  num_workers=0,
                                  prefetch_factor=prefetch,
                                  pin_memory=no_pagemem)
    for X, y in test_dataloader:
        print(f"Shape e tipo dei campioni: {X.shape}, {X.dtype}")
        print(f"Shape e tipo delle etichette: {y.shape} {y.dtype}")
        break
    return (train_dataloader, val_dataloader, test_dataloader)

train_loader, val_loader, test_loader = make_dataloaders(train_data, val_data, test_data)
```

---

## PASSO 8 — `eval_loop`

Copiare da `torchnn.py` e aggiungere `y_score`. Attenzione al `torch.exp(pred)`:

```python
def eval_loop(model, dataloader, device,
              loss_fn=config["test_loss"],
              metrics=config["metrics"],
              average=config["metric_average"]):

    model.eval()
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, accuracy = 0.0, 0

    y_true  = []
    y_pred  = []
    y_score = []      # ← AGGIUNGERE

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred  = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy  += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.argmax(1).cpu().numpy())

            # ← AGGIUNGERE — scegli in base al task:
            y_score.extend(torch.exp(pred).cpu().numpy())         # multiclasse
            # y_score.extend(torch.exp(pred)[:, 1].detach().tolist()) # binario (prob. classe positiva)

    test_loss /= num_batches
    accuracy  /= size

    epoch_metrics = {}
    for metric in metrics:
        epoch_metrics[metric.__name__] = metric(y_true, y_pred, average=average) \
                if 'average' in signature(metric).parameters.keys() \
                else metric(y_true, y_pred)

    return test_loss, accuracy, epoch_metrics, y_true, y_pred, y_score  # ← return aggiornato
```

---

## PASSO 9 — `EarlyStopping` e `train_loop`

Copiare entrambe da `torchnn.py` **senza nessuna modifica**.

---

## PASSO 10 — `train_test`, `save_model`, `load_model`

Copiare il blocco da `torchnn.py` con le seguenti modifiche a `train_test`:

### Modifiche a `train_test`

**A. Aggiungere `best_validate_metric` in cima al corpo della funzione:**

```python
# Per salvare il modello con validation loss minima:
best_validate_metric = float("inf")

# OPPURE per salvare il modello con validation accuracy massima:
# best_validate_metric = float("-inf")
```

**B. Aggiornare l'unpacking del val_dataloader:**

```python
# Originale:
epoch_validate_loss, _, _ = eval_loop(...)
# → Sostituire con:
epoch_validate_loss, epoch_validate_accuracy, epoch_validate_metrics, *_ = eval_loop(
    model, val_dataloader, device, loss_fn=test_loss_fn, metrics=metrics, average=average
)
```

**C. Aggiornare l'unpacking del test_dataloader:**

```python
# Originale:
epoch_test_loss, epoch_accuracy, epoch_metrics = eval_loop(...)
# → Sostituire con:
epoch_test_loss, epoch_accuracy, epoch_metrics, *_ = eval_loop(
    model, test_dataloader, device, loss_fn=test_loss_fn, metrics=metrics, average=average
)
```

**D. Aggiungere il salvataggio del modello migliore prima del blocco early stopping:**

```python
# Criterio: minimizzazione validation loss
if val_dataloader is not None:
    if epoch_validate_loss < best_validate_metric:
        best_validate_metric = epoch_validate_loss
        save_model(model, optimizer, epoch,
                   train_loss, validation_loss, test_loss,
                   accuracy, test_metrics,
                   "best_model.pth")   # ← percorso del file

# OPPURE criterio: massimizzazione validation accuracy
# if val_dataloader is not None:
#     if epoch_validate_accuracy > best_validate_metric:
#         best_validate_metric = epoch_validate_accuracy
#         save_model(model, optimizer, epoch,
#                    train_loss, validation_loss, test_loss,
#                    accuracy, test_metrics,
#                    "best_model.pth")
```

> ⚠️ `save_model` ha questa firma:
> ```python
> save_model(model, optimizer, epoch,
>            train_loss, validation_loss, test_loss,
>            accuracy, test_metrics, path)
> ```
> Gli argomenti `train_loss`, `validation_loss`, `test_loss`, `accuracy`, `test_metrics`
> sono le **liste** accumulate fino all'epoca corrente — coincidono con i valori
> del return di `train_test`. Passarle direttamente.

---

## PASSO 11 — `input_dim` e `output_dim`

```python
input_dim  = X_tr.shape[1]      # numero di feature
output_dim = len(set(y_tr))     # numero di classi (es. 2 per binario, N per multiclasse)
```

---

## PASSO 12 — Architettura della rete

Scegliere **una** delle due architetture. `LogSoftmax(dim=1)` e niente `squeeze`
valgono in entrambi i casi di classificazione.

### DNN (rete densa)

```python
class ReteDensa(nn.Module):
    def __init__(self, input_dim, output_dim):
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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, output_dim),    # output_dim = 2 (binario) o N (multiclasse)
            nn.LogSoftmax(dim=1)          # sempre uguale per classificazione
        )
    def forward(self, x):
        return self.network(x)            # niente squeeze per classificazione
```

### CNN (rete convoluzionale Conv1d)

Con padding "same" (`padding = (kernel_size-1)//2`) la lunghezza resta invariata
dopo ogni Conv1d. Solo il MaxPool dimezza: `flat_dim = out_channels * (input_dim // 2)`.
Su questa parte non ho fatto esercitazioni, approfondisci tu per avere contezza di come diminuisce la dimensionalità dei batch.

```python
class ReteConv1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReteConv1d, self).__init__()

        flat_dim = 64 * (input_dim // 2)   # ← aggiorna se cambi out_channels o n. di MaxPool

        self.network = nn.Sequential(
            # blocco convoluzionale
            nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # o AvgPool1d(2)
            # blocco denso
            nn.Flatten(),
            nn.Linear(flat_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, output_dim),     # output_dim = 2 (binario) o N (multiclasse)
            nn.LogSoftmax(dim=1)           # sempre uguale per classificazione
        )
    def forward(self, x):
        x = x.unsqueeze(1)                 # [batch, F] → [batch, 1, F]
        return self.network(x)             # niente squeeze per classificazione
```

> **Metodo dummy per verificare `flat_dim`** (usa se hai dubbi):
> ```python
> with torch.no_grad():
>     dummy    = torch.zeros(1, 1, input_dim)
>     flat_dim = nn.Sequential(*list(model.network.children())[:8])(dummy).view(1,-1).shape[1]
> print(flat_dim)
> ```

---

## PASSO 13 — Istanzia il modello e controlla i parametri

```python
from torchinfo import summary

model = ReteDensa(input_dim, output_dim)   # oppure ReteConv1d(input_dim, output_dim)
summary(model)
```

---

## PASSO 14 — `EarlyStopping`

```python
es = EarlyStopping()
```

---

## PASSO 15 — Ottimizzatore

```python
from torch.optim import Adam   # oppure SGD, Adagrad, ecc.

opt = Adam(model.parameters(),
           lr=config["learning_rate"],
           weight_decay=0.01)    # ← modifica o rimuovi in base alla richiesta
```

---

## PASSO 16 — Scheduler (solo se richiesto)

```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(opt, gamma=0.9)
```

> Se non richiesto, passa `scheduler=None` a `train_test` (è il default).

---

## PASSO 17 — Lancia `train_test`

```python
train_loss, val_loss, test_loss, accuracy, metrics = train_test(
    model            = model,
    optimizer        = opt,
    device           = device,
    train_dataloader = train_loader,
    test_dataloader  = test_loader,
    val_dataloader   = val_loader,
    early_stopping   = es,
    scheduler        = scheduler    # rimuovere se non usato
)
```

---

## PASSO 18 — Display (solo se richiesto)

```python
from torchnn import displayLosses, displayMetrics   # oppure sono già nel notebook

displayLosses(train_loss, test_loss, val_loss)
displayMetrics(accuracy, metrics)
```

---

## PASSO 19 — Carica il modello migliore

Gli argomenti da creare per il return di `load_model` corrispondono esattamente
a quelli del return di `train_test`:

```python
model, opt, checkpoint = load_model("best_model.pth", model, opt, device)
```

---

## PASSO 20 — `eval_loop` sul modello migliore

```python
test_loss, accuracy, epoch_metrics, y_true_nn, y_pred_nn, y_score_nn = eval_loop(
    model, test_loader, device
)
```

---

## PASSO 21 — Metriche e ROC

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_true_nn, y_pred_nn)
print(conf)
```

### ROC AUC

```python
import numpy as np
from sklearn.metrics import roc_auc_score

y_score_nn_arr = np.array(y_score_nn)   # lista → array [n_campioni, n_classi]

# Binario (probabilità classe positiva)
roc_auc_score(y_true_nn, y_score_nn_arr[:, 1])

# Multiclasse One-vs-Rest
roc_auc_score(y_true_nn, y_score_nn_arr, multi_class="ovr", average="macro")
```