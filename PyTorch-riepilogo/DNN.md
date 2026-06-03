# Checklist Esame — Rete Neurale con `torchnn.py`

> Segui **l'ordine dei passi** riportato qui: alcune modifiche a `torchnn.py` vanno fatte
> prima ancora di scrivere la rete, perché cambiano le signature delle funzioni usate dopo.

---

## FASE 0 — Preprocessing (nel notebook, prima di tutto)

Passi standard identici a qualsiasi modello sklearn:

1. `train_test_split` (doppio se serve val set: train/val/test)
2. `StandardScaler` → `fit_transform` su train, `transform` su val e test
3. Selezione feature (`SelectKBest`, correlazione, ecc.)
4. Verifica shape: `X_tr.shape`, `len(set(y_tr))`

---

## FASE 1 — Modifiche a `torchnn.py` (prima di scrivere qualsiasi altra cosa)

Apri subito `torchnn.py` e fai tutte le modifiche necessarie. L'ordine qui sotto
rispecchia l'ordine in cui le funzioni compaiono nel file.

### 1a. Import iniziali — adatta i pacchetti al task

In cima a `torchnn.py` gli import di default sono:

```python
from sklearn.metrics import precision_score, recall_score, f1_score
```

Nel caso di **regressione** queste metriche non hanno senso. Sostituirle con:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

| Task | Metriche |
|---|---|
| Classificazione | `precision_score, recall_score, f1_score` |
| Regressione | `mean_absolute_error, mean_squared_error, r2_score` |

> Le metriche di regressione **non hanno il parametro `average`**, ma `eval_loop` gestisce
> già questo caso con il ramo `else metric(y_true, y_pred)` — nessuna altra modifica necessaria.

---

### 1b. `config` — adatta loss, metriche e metric_average

```python
config = {
    "learning_rate": ...,
    "batch_size": 64,
    "epochs": 20,
    "patience": 5,
    "min_delta": 0.01,
    "train_loss": ...,         # ← vedi tabella
    "test_loss":  ...,         # ← vedi tabella
    "metrics": [...],          # ← vedi tabella import sopra
    "metric_average": 'macro'  # per regressione: None (ignorato comunque)
}
```

| Task | `train_loss` | `test_loss` |
|---|---|---|
| Classificazione binaria/multiclasse | `nn.NLLLoss()` | `nn.NLLLoss()` |
| Regressione | `nn.MSELoss()` | `nn.MSELoss()` |

> ⚠️ Il `torchnn.py` fornito ha `CrossEntropyLoss` come `train_loss` di default.
> Se usi `LogSoftmax` nell'ultimo layer **devi cambiarlo in `NLLLoss`** — sono incompatibili.

---

### 1c. `eval_loop` — aggiungere `y_score` e (per regressione) rimuovere accuracy

`eval_loop` è la seconda funzione da modificare perché il suo return viene usato
sia dentro `train_test` che nella valutazione finale.

#### Modifiche per la curva ROC (classificazione)

Il `torchnn.py` originale **non raccoglie `y_score`**. Aggiungere:

```python
# 1. Nella sezione di inizializzazione delle liste, dopo y_pred = []:
y_score = []

# 2. Nel ciclo with torch.no_grad(), dopo y_pred.extend(...):
y_score.extend(torch.exp(pred).detach().tolist())
# torch.exp() inverte il LogSoftmax → probabilità in [0,1]

# 3. Nel return, aggiungere y_score in coda:
return test_loss, accuracy, epoch_metrics, y_true, y_pred, y_score
```

#### Modifiche per la regressione

Nel caso di regressione `accuracy` non ha senso. Va rimossa o sostituita con
una metrica continua (es. MAE). Le occorrenze da modificare in `eval_loop` sono:

```python
# RIMUOVERE o commentare:
accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
# ...
accuracy /= size

# Nel return, sostituire accuracy con 0.0 o una metrica continua:
return test_loss, 0.0, epoch_metrics, y_true, y_pred, y_score
```

---

### 1d. `train_test` — aggiornare gli unpacking di `eval_loop` e aggiungere il checkpoint

Dopo aver modificato il return di `eval_loop`, aggiornare le due righe in
`train_test` dove viene chiamato:

```python
# Validation (riga originale):
epoch_validate_loss, _, _ = eval_loop(...)
# → aggiornare a:
epoch_validate_loss, epoch_validate_acc, _, *_ = eval_loop(...)

# Test (riga originale):
epoch_test_loss, epoch_accuracy, epoch_metrics = eval_loop(...)
# → aggiornare a:
epoch_test_loss, epoch_accuracy, epoch_metrics, *_ = eval_loop(...)
```

> `*_` cattura y_true, y_pred, y_score che durante il training non servono.

#### Aggiungere il checkpoint del miglior modello

Nel blocco `train_test`, subito **prima** del blocco `early_stopping`,
aggiungere la logica di salvataggio. Scegliere uno dei due criteri:

**Criterio A — massimizzazione della validation accuracy:**

```python
# Aggiungere nella sezione delle variabili iniziali di train_test (in cima al corpo):
best_val_acc = float("-inf")

# Aggiungere nel ciclo epoche, dopo aver calcolato epoch_validate_acc:
if val_dataloader is not None:
    if epoch_validate_acc > best_val_acc:
        best_val_acc = epoch_validate_acc
        save_model(model, optimizer, epoch,
                   train_loss, validation_loss, test_loss,
                   accuracy, test_metrics, "best_model.pth")
```

**Criterio B — minimizzazione della validation loss:**

```python
# Aggiungere nella sezione delle variabili iniziali di train_test:
best_val_loss = float("inf")

# Aggiungere nel ciclo epoche, dopo aver calcolato epoch_validate_loss:
if val_dataloader is not None:
    if epoch_validate_loss < best_val_loss:
        best_val_loss = epoch_validate_loss
        save_model(model, optimizer, epoch,
                   train_loss, validation_loss, test_loss,
                   accuracy, test_metrics, "best_model.pth")
```

> ⚠️ `save_model` riceve le **liste fino all'epoca corrente**, non scalari.
> Viene chiamata dentro il ciclo epoche, quindi le liste crescono ad ogni epoch
> e il checkpoint salva sempre lo storico completo fino a quel momento.

> ⚠️ `save_model` accetta anche `val_loss` come argomento: se non usi un val set
> puoi passare una lista vuota `[]`.

#### Regressione: accuracy in `train_test`

In `train_test` la variabile `accuracy` viene accumulata e restituita.
Nel caso di regressione va rinominata o ignorata. Le occorrenze sono:

```python
# Riga da modificare nel ciclo epoche:
accuracy.append(epoch_accuracy)   # epoch_accuracy sarà 0.0 se modificato in eval_loop

# Nella print:
print(f"... Accuracy: {epoch_accuracy:6.2f} ...")
# → sostituire con una metrica significativa o rimuovere dalla stampa
```

#### Regressione: `displayMetrics` e scala dell'asse y

`displayMetrics` ha `plt.ylim(0, 1)` pensato per metriche in [0,1] come precision/recall/f1.
Per regressione MAE, MSE e R² possono avere range molto diversi: rimuovere quella riga.

```python
# In displayMetrics, rimuovere o commentare:
plt.ylim(0, 1)
```

---

## FASE 2 — Nel notebook: `config` e `TensDataset`

### 2a. Sovrascrivere `config` se necessario

```python
config["learning_rate"] = 0.01
config["epochs"] = 20
# ecc.
```

### 2b. Classe `TensDataset` (non è in `torchnn.py`, va scritta nel notebook)

```python
class TensDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.long)    # classificazione
        # self.y = torch.tensor(y_data.values, dtype=torch.float32) # regressione
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

| Task | dtype di `y` |
|---|---|
| Classificazione | `torch.long` |
| Regressione | `torch.float32` |

---

## FASE 3 — DataLoader

```python
train_data = TensDataset(X_tr, y_tr)
val_data   = TensDataset(X_val, y_val)
test_data  = TensDataset(X_te, y_te)

# ⚠️ In locale: num_workers=0 e prefetch_factor=None per evitare errori
train_loader, val_loader, test_loader = make_dataloaders(
    train_data, val_data, test_data, batch=config["batch_size"]
)
```

> ⚠️ Il `torchnn.py` fornito ha `num_workers=2` e `prefetch_factor=12` come default.
> Se dà errori in locale, sovrascrivere con `num_workers=0` e `prefetch_factor=None`.

---

## FASE 4 — Architettura della rete

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
            nn.Linear(16, N_OUT),      # ← vedi tabella
            ATTIVAZIONE_FINALE         # ← vedi tabella
        )
    def forward(self, x):
        return self.network(x).squeeze(1)  # utile solo se N_OUT = 1
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

## FASE 5 — Ottimizzatore, Scheduler, EarlyStopping

```python
# SGD con momentum e Nesterov
opt = torch.optim.SGD(model.parameters(),
                      lr=config["learning_rate"],
                      momentum=config["momentum"],
                      nesterov=config["nesterov"])

# oppure Adagrad con weight decay
opt = torch.optim.Adagrad(model.parameters(),
                          lr=config["learning_rate"],
                          weight_decay=0.01)

# Scheduler (opzionale)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.90)

# Early stopping
es = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
```

---

## FASE 6 — Training

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

displayLosses(train_loss, test_loss, val_loss)
displayMetrics(accuracy, metrics)
```

---

## FASE 7 — Valutazione finale

```python
# Caricare il miglior checkpoint salvato durante il training
model, opt, checkpoint = load_model("best_model.pth", model, opt, device)

# Eseguire eval_loop sul test set per ottenere y_true, y_pred, y_score
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
import numpy as np
from sklearn.metrics import roc_auc_score

y_score_nn_arr = np.array(y_score_nn)  # lista di liste → array [n_campioni, n_classi]

# Classificazione binaria (probabilità della classe positiva)
roc_auc_score(y_true_nn, y_score_nn_arr[:, 1])

# Classificazione multiclasse One-vs-Rest
roc_auc_score(y_true_nn, y_score_nn_arr, multi_class="ovr", average="macro")
```

### Curva ROC (One-vs-Rest)

```python
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

lb = LabelBinarizer()
y_bin = lb.fit(y_tr).transform(y_true_nn)  # fit su train, transform su test

fig, ax = plt.subplots(figsize=(8, 6))
for classe, y_bin_col, y_score_col in zip(lb.classes_, y_bin.T, y_score_nn_arr.T):
    RocCurveDisplay.from_predictions(
        y_bin_col, y_score_col,
        name=f"ROC Classe {classe}",
        ax=ax, despine=True
    )
plt.title("Curve ROC One-vs-Rest")
plt.legend()
plt.show()
```

---

## Riepilogo delle modifiche a `torchnn.py` (in ordine di apparizione nel file)

| # | Dove | Cosa modificare | Quando serve |
|---|---|---|---|
| 1 | Import iniziali | `precision_score, recall_score, f1_score` → metriche di regressione | Regressione |
| 2 | `config` | `CrossEntropyLoss` → `NLLLoss` | Se usi `LogSoftmax` |
| 3 | `config` | `MSELoss` per train e test + metriche regressione + `metric_average=None` | Regressione |
| 4 | `eval_loop` | Aggiungere `y_score` e modificare `return` | Sempre (per ROC) |
| 5 | `eval_loop` | Rimuovere/sostituire `accuracy` | Regressione |
| 6 | `train_test` | Aggiornare unpacking val: `epoch_validate_loss, epoch_validate_acc, _, *_` | Se modifichi `eval_loop` |
| 7 | `train_test` | Aggiornare unpacking test: `epoch_test_loss, epoch_accuracy, epoch_metrics, *_` | Se modifichi `eval_loop` |
| 8 | `train_test` | Aggiungere `best_val_acc = float("-inf")` o `best_val_loss = float("inf")` in cima al corpo | Se richiesto checkpoint |
| 9 | `train_test` | Aggiungere blocco `save_model(...)` prima dell'early stopping | Se richiesto checkpoint |
| 10 | `displayMetrics` | Rimuovere `plt.ylim(0, 1)` | Regressione |
| 11 | `make_dataloaders` | `num_workers=0`, `prefetch_factor=None` | In locale |
