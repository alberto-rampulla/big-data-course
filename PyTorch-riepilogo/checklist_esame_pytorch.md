# Checklist Esame — Rete Neurale con `torchnn.py`

## 0. Preprocessing (prima della rete)

Passi standard **identici a qualsiasi modello sklearn**:

1. `train_test_split` (eventualmente doppio: train/val/test)
2. `StandardScaler` → `fit_transform` su train, `transform` su val e test
3. Selezione feature (`SelectKBest`, correlazione, ecc.)
4. Verifica shape: `X_tr.shape`, `len(set(y_tr))`

---

## 1. Configurazione (`config`)

```python
config = {
    "learning_rate": ...,
    "batch_size": 64,
    "epochs": 20,
    "patience": 5,
    "min_delta": 0.01,
    "train_loss": ...,   # ← vedi tabella sotto
    "test_loss":  ...,   # ← vedi tabella sotto
    "metrics": [precision_score, recall_score, f1_score],
    "metric_average": 'macro'
}
```

| Task | `train_loss` | `test_loss` |
|---|---|---|
| Classificazione binaria/multiclasse | `nn.NLLLoss()` | `nn.NLLLoss()` |
| Regressione | `nn.MSELoss()` | `nn.MSELoss()` |

> ⚠️ **Attenzione:** il `torchnn.py` fornito ha `CrossEntropyLoss` come `train_loss` di default.
> Se usi `LogSoftmax` nell'ultimo layer, **devi cambiarlo in `NLLLoss`** — le due non sono compatibili insieme.

---

## 2. Dataset e DataLoader

### 2a. Classe `TensDataset`

Da scrivere nel notebook (non è in `torchnn.py`):

```python
class TensDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.long)   # long per classificazione
        # self.y = torch.tensor(y_data.values, dtype=torch.float32)  # float per regressione
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

| Task | dtype di `y` |
|---|---|
| Classificazione | `torch.long` |
| Regressione | `torch.float32` |

### 2b. DataLoader

```python
train_data = TensDataset(X_tr, y_tr)
val_data   = TensDataset(X_val, y_val)
test_data  = TensDataset(X_te, y_te)

# ⚠️ In locale usare num_workers=0 e prefetch=None per evitare errori
train_loader, val_loader, test_loader = make_dataloaders(
    train_data, val_data, test_data, batch=config["batch_size"]
)
```

> ⚠️ `make_dataloaders` in `torchnn.py` ha `num_workers=2` e `prefetch_factor=12` come default.
> In locale (o se dà errori) sovrascrivere con `num_workers=0` e `prefetch_factor=None`.

---

## 3. Architettura della rete

```python
class ReteDensa(nn.Module):
    def __init__(self, input_dim):
        super(ReteDensa, self).__init__()
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
            nn.Linear(16, N_OUT),   # ← vedi tabella sotto
            ATTIVAZIONE_FINALE      # ← vedi tabella sotto
        )
    def forward(self, x):
        return self.network(x).squeeze(1)  # squeeze utile solo se N_OUT = 1
```

| Task | `N_OUT` | Attivazione finale | `squeeze(1)` |
|---|---|---|---|
| Classificazione binaria | `2` | `nn.LogSoftmax(dim=1)` | ❌ inutile |
| Classificazione multiclasse | `N` (n. classi) | `nn.LogSoftmax(dim=1)` | ❌ inutile |
| Classificazione binaria (alt.) | `1` | `nn.LogSigmoid()` | ✅ necessario |
| Regressione | `1` | nessuna | ✅ necessario |

```python
device    = "cpu"
input_dim = X_tr.shape[1]
model     = ReteDensa(input_dim).to(device)
```

---

## 4. Ottimizzatore e Scheduler

```python
# Esempi di ottimizzatori
opt = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                      momentum=config["momentum"], nesterov=config["nesterov"])

opt = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)

# Scheduler (opzionale)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.90)

# Early stopping
es = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
```

---

## 5. Training

```python
train_loss, val_loss, test_loss, accuracy, metrics = train_test(
    model=model,
    optimizer=opt,
    scheduler=scheduler,
    early_stopping=es,
    device=device,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    val_dataloader=val_loader
)
```

```python
displayLosses(train_loss, test_loss, val_loss)
displayMetrics(accuracy, metrics)
```

---

## 6. Modifiche a `eval_loop` per la ROC

Il `eval_loop` originale **non restituisce `y_score`** (le probabilità).
Per poter fare la curva ROC, aggiungere `y_score` manualmente:

```python
# Aggiungere nella inizializzazione delle liste in eval_loop:
y_score = []

# Aggiungere nel ciclo with torch.no_grad(), dopo y_pred.extend(...):
y_score.extend(torch.exp(pred).detach().tolist())
# torch.exp() inverte il LogSoftmax → probabilità vere [0,1]

# Aggiungere nel return:
return test_loss, accuracy, epoch_metrics, y_true, y_pred, y_score
```

> ⚠️ Se modifichi il return di `eval_loop`, aggiorna anche le chiamate in `train_test`:
> ```python
> epoch_validate_loss, _, _, *_ = eval_loop(...)   # val
> epoch_test_loss, epoch_accuracy, epoch_metrics, *_ = eval_loop(...)  # test
> ```

---

## 7. Valutazione finale (dopo il training)

```python
# Caricare il miglior modello se salvato
model, opt, checkpoint = load_model("best_model.pth", model, opt, device)

# Eseguire eval_loop sul test set
test_loss, accuracy, epoch_metrics, y_true_nn, y_pred_nn, y_score_nn = eval_loop(
    model, test_loader, device
)
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_true_nn, y_pred_nn)
```

### ROC AUC
```python
from sklearn.metrics import roc_auc_score
import numpy as np

y_score_nn_arr = np.array(y_score_nn)   # lista di liste → array [n_campioni, n_classi]

# Classificazione binaria
roc_auc_score(y_true_nn, y_score_nn_arr[:, 1])

# Classificazione multiclasse (One-vs-Rest)
roc_auc_score(y_true_nn, y_score_nn_arr, multi_class="ovr", average="macro")
```

### Curva ROC (One-vs-Rest)
```python
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

lb = LabelBinarizer()
y_bin = lb.fit(y_tr).transform(y_true_nn)   # fit su train, transform su test

fig, ax = plt.subplots(figsize=(8, 6))
for classe, y_bin_col, y_score_col in zip(lb.classes_, y_bin.T, y_score_nn_arr.T):
    RocCurveDisplay.from_predictions(
        y_bin_col, y_score_col, name=f"ROC Classe {classe}", ax=ax, despine=True
    )
plt.title("Curve ROC One-vs-Rest")
plt.legend()
plt.show()
```

---

## Riepilogo rapido delle modifiche a `torchnn.py`

| Cosa | Modifica necessaria |
|---|---|
| `config["train_loss"]` | `CrossEntropyLoss` → `NLLLoss` se usi `LogSoftmax` |
| `make_dataloaders` | `num_workers=0`, `prefetch_factor=None` se in locale |
| `eval_loop` return | Aggiungere `y_score` per la curva ROC |
| `train_test` unpacking | Aggiornare se `eval_loop` restituisce più valori |
