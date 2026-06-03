# Riepilogo: `squeeze(1)` nei task di PyTorch
 
## Quando usare `squeeze(1)` nell'ultimo layer?
 
| Task | Loss | Ultimo layer | `squeeze(1)` |
|---|---|---|---|
| Classificazione binaria | `NLLLoss`+ `LogSoftmax` | `Linear(16, 2)` | ❌ Inutile |
| Classificazione multiclasse | `NLLLoss` + `LogSoftmax` | `Linear(16, N)` | ❌ Inutile |
| Classificazione binaria | `BCELoss` + `LogSigmoid` | `Linear(16, 1)` | ✅ Necessario |
| Regressione | `MSELoss` | `Linear(16, 1)` | ✅ Necessario |
 
## Regola pratica
 
> `squeeze(1)` serve ogni volta che l'ultimo layer ha **1 solo neurone**,
> ovvero quando l'output ha shape `[batch_size, 1]` e devi ridurlo a `[batch_size]`.

# Riepilogo: Task, Loss, Attivazione e Neuroni in PyTorch
 
| Task | Train Loss | Test Loss | Attivazione finale | Neuroni ultimo layer |
|---|---|---|---|---|
| Classificazione binaria | `NLLLoss` | `NLLLoss` | `LogSoftmax(dim=1)` | 2 |
| Classificazione binaria (alternativa) | `BCELoss` | `BCELoss` | `LogSigmoid` | 1 |
| Classificazione multiclasse | `NLLLoss` | `NLLLoss` | `LogSoftmax(dim=1)` | N (numero di classi) `len(set(y_tr))` |
| Regressione | `MSELoss` | `MSELoss` | Nessuna | 1 |
 
## Perché Train Loss e Test Loss sono uguali?
 
Durante il **training**, la loss viene usata per calcolare i gradienti e aggiornare i pesi.
Durante il **test**, la loss viene usata solo per **valutare** le prestazioni del modello (nessun aggiornamento dei pesi).