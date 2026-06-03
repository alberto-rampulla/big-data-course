# Checklist Esame — Rete Convoluzionale `Conv1d` su dati tabulari

> Le fasi di **preprocessing**, **config**, **DataLoader**, **training**, **valutazione** e le
> **modifiche a `torchnn.py`** sono identiche alla checklist delle reti dense.
> Questo documento si concentra sulle **differenze** introdotte da `Conv1d`.

---

## Differenza concettuale rispetto alla rete densa

Con dati tabulari multidimensionali, ogni campione è un vettore di `F` feature.
`Conv1d` lo tratta come una **sequenza di F "passi"**, ognuno con `C` canali.

```
Rete densa:   input shape → [batch, F]
Conv1d:       input shape → [batch, C_in, F]   ← serve un reshape nel forward!
```

Per dati tabulari si parte tipicamente con **1 solo canale** (`C_in = 1`):
ogni feature è un "passo temporale" di 1 canale.

---

## 1. `TensDataset` — nessuna modifica

Il dataset rimane identico: restituisce tensori `[F]` per ogni campione.
Il reshape a `[C_in, F]` si fa **dentro il `forward`**, non nel dataset.

```python
class TensDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.long)    # long per classificazione
        # self.y = torch.tensor(y_data.values, dtype=torch.float32) # float per regressione
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

---

## 2. Architettura `Conv1d`

```python
class ReteConv1d(nn.Module):
    def __init__(self, input_dim, n_classi):
        super(ReteConv1d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # dimezza la lunghezza della sequenza
        )

        flat_dim = 32 * (input_dim // 2)   # ← vedi regola sotto

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classi),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)      # [batch, F] → [batch, 1, F]: aggiunge il canale
        x = self.conv(x)        # [batch, 32, F//2]
        x = self.fc(x)          # Flatten → [batch, 32*(F//2)] → ... → [batch, n_classi]
        return x
```

```python
device    = "cpu"
input_dim = X_tr.shape[1]       # numero di feature
n_classi  = len(set(y_tr))
model     = ReteConv1d(input_dim, n_classi).to(device)
```

---

## 3. Regola per calcolare `flat_dim` (dimensione dopo il Flatten)

La dimensione in ingresso al primo `Linear` dopo il `Flatten` dipende da:
- **`out_channels`** dell'ultimo layer Conv1d
- **lunghezza della sequenza** dopo tutti i layer Conv/Pool

### Formula generale dopo ogni layer

**Dopo `Conv1d`:**

$$L_{out} = \left\lfloor \frac{L_{in} + 2 \cdot padding - kernel\_size}{stride} \right\rfloor + 1$$

Con `padding=1`, `kernel_size=3`, `stride=1` (default):

$$L_{out} = \frac{L_{in} + 2 - 3}{1} + 1 = L_{in} \quad \text{(lunghezza invariata)}$$

**Dopo `MaxPool1d(kernel_size=2)`:**

$$L_{out} = \left\lfloor \frac{L_{in}}{2} \right\rfloor$$

### Esempio con `input_dim = 6`, architettura sopra

```
Input:            [batch,  1,  6]
Conv1d(1→16):     [batch, 16,  6]   padding=1, kernel=3 → lunghezza invariata
Conv1d(16→32):    [batch, 32,  6]   padding=1, kernel=3 → lunghezza invariata
MaxPool1d(2):     [batch, 32,  3]   6 // 2 = 3
Flatten:          [batch, 96]       32 × 3 = 96
```

Quindi: `flat_dim = 32 * (6 // 2) = 96`

### Regola pratica da ricordare

> Con **`padding = (kernel_size - 1) / 2`** (padding "same"), ogni `Conv1d`
> **non cambia la lunghezza**. Solo `MaxPool1d` la riduce.
>
> Se usi **un solo** `MaxPool1d(kernel_size=2)`:
> ```python
> flat_dim = out_channels_ultimo_conv * (input_dim // 2)
> ```
>
> Se usi **due** `MaxPool1d(kernel_size=2)`:
> ```python
> flat_dim = out_channels_ultimo_conv * (input_dim // 4)
> ```
>
> In generale: ogni `MaxPool1d(2)` dimezza, ogni `MaxPool1d(4)` divide per 4.

### Metodo infallibile per non sbagliare mai

Se l'architettura è complessa, calcola `flat_dim` in modo automatico con un passaggio dummy:

```python
with torch.no_grad():
    dummy = torch.zeros(1, 1, input_dim)   # [1, C_in, F]
    out   = conv_block(dummy)              # passa solo il blocco conv
    flat_dim = out.view(1, -1).shape[1]    # conta gli elementi dopo flatten
print(f"flat_dim = {flat_dim}")
```

---

## 4. Confronto densa vs Conv1d

| | Rete Densa | Rete Conv1d |
|---|---|---|
| Input shape nel `forward` | `[batch, F]` | `[batch, 1, F]` (dopo `unsqueeze(1)`) |
| `unsqueeze(1)` nel `forward` | ❌ | ✅ necessario |
| `TensDataset` | invariato | invariato |
| `flat_dim` | non esiste | `out_ch × L_finale` |
| `Flatten` | non necessario | ✅ necessario prima dei layer FC |
| Resto (`config`, `train_test`, `eval_loop`, ROC) | — | identico |

---

## 5. Errori comuni

| Errore | Causa | Soluzione |
|---|---|---|
| `RuntimeError: Expected 3D input` | Manca `unsqueeze(1)` nel `forward` | Aggiungere `x = x.unsqueeze(1)` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | `flat_dim` calcolato male | Usare il metodo dummy |
| `flat_dim` sbagliato con `input_dim` dispari | `//` tronca: `5 // 2 = 2` non `2.5` | Usare sempre `//` (floor division) |
