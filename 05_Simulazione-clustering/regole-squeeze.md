# Riepilogo: `squeeze(1)` nei task di PyTorch
 
## Quando usare `squeeze(1)` nell'ultimo layer?
 
| Task | Loss | Ultimo layer | `squeeze(1)` |
|---|---|---|---|
| Classificazione binaria | `NLLLoss` | `Linear(16, 2)` | ❌ Inutile |
| Classificazione multiclasse | `NLLLoss` | `Linear(16, N)` | ❌ Inutile |
| Classificazione binaria | `BCELoss` + `LogSigmoid` | `Linear(16, 1)` | ✅ Necessario |
| Regressione | `MSELoss` | `Linear(16, 1)` | ✅ Necessario |
 
## Regola pratica
 
> `squeeze(1)` serve ogni volta che l'ultimo layer ha **1 solo neurone**,
> ovvero quando l'output ha shape `[batch_size, 1]` e devi ridurlo a `[batch_size]`.
 