# Riepilogo ML con scikit-learn — Tutto il pre-reti neurali

> Basato su: `01_Preprocessing`, `02_Capacità`, `03_Clustering`, `04_Classificazione` (binaria, multiclasse, regressione)

---

## 0. La pipeline mentale (checklist da seguire SEMPRE)

Questo è l'ordine corretto delle operazioni in qualunque esercizio di preprocessing + modello:

```
0. EDA (shape, describe, isna)
1. Rimozione colonne/righe troppo sparse (NA)
2. Rimozione feature "prognostiche" / leakage (se presenti nel testo dell'esercizio)
3. Split train/test (stratify=y se classificazione)
   └── DA QUI IN POI: fit_transform SOLO su train, transform su test
4. Imputazione dei missing (SimpleImputer)
5. Encoding delle categoriche (OrdinalEncoder / OneHotEncoder)
6. Standardizzazione (StandardScaler) — mai sul target in regressione!
7. Analisi di correlazione (con target + tra feature → multicollinearità)
8. Feature Selection (SelectKBest / SelectFromModel)
9. Fit del modello (+ eventuale GridSearchCV) e valutazione con le metriche adeguate
```

**Regola d'oro anti data-leakage:** tutto ciò che "impara" parametri dai dati (medie, deviazioni standard, categorie, soglie...) si fa con `fit_transform` **solo sul training set**; sul test set si usa solo `transform`. Per questo lo split va fatto **il prima possibile**, subito dopo l'EDA/pulizia sparsità.

---

## 1. Preprocessing

### 1.1 Import e caricamento dati

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn._config import set_config   
set_config(transform_output='pandas')   # fa restituire DataFrame invece di array numpy

dd = pd.read_csv('data/file.csv', sep=';', index_col=0) #stare attenti ai separatori, alcune volte serve altre no
dd.shape
dd.describe()
dd.info()          # alternativa rapida: dtype + n° non-null per colonna in un colpo solo
```

### 1.2 Analisi dati mancanti e rimozione colonne/righe sparse

```python
dd.isna().sum() / len(dd)              # % NA per colonna
```

**Nota — colonne ad alta cardinalità / non informative:** oltre alla sparsità, è buona norma eliminare *fin da subito* le colonne con troppe modalità uniche o puramente identificative (es. `PassengerId`, `Name`, `Ticket`, ID/codici testuali liberi). Non sono missing, ma non aiutano il modello e l'encoding le tratterebbe in modo inefficiente.

```python
dati = dati.drop(columns=["Ticket", "Name", "Cabin", "PassengerId"])
```

### 1.3 Rimozione feature "prognostiche" (data leakage concettuale)

Non solo leakage train/test: anche usare feature che derivano dal target stesso (es. score clinici calcolati dopo l'evento) è leakage. Vanno rimosse **prima** di tutto, in base al dominio/consegna.

### 1.4 Split train/test 

```python
from sklearn.model_selection import train_test_split

X = dd.drop(columns=['target'])
y = dd['target']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42   # stratify SOLO se classificazione
)
```

**Nota — stratify in regressione:** se il target è continuo non si può fare `stratify=y`, ma se nel dataset c'è una **feature categorica importante e sbilanciata** (es. `ocean_proximity`, con qualche modalità rara), conviene comunque stratificare lo split su quella, per essere sicuri che tutte le categorie compaiano sia in train che in test:

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=123,
                                          stratify=dati["ocean_proximity"])
```

### 1.5 Imputazione missing (DOPO lo split)

```python
from sklearn.impute import SimpleImputer

# divido le var numeriche da categoriali perche hanno metodi di imputazioni diversi 
num_cols = X_tr.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X_tr.select_dtypes(include='object').columns

# Imputatori
imp_num = SimpleImputer(strategy='median')          # o 'mean'
imp_cat = SimpleImputer(strategy='most_frequent')    # o 'constant', fill_value='missing'

X_tr[num_cols] = imp_num.fit_transform(X_tr[num_cols])
X_te[num_cols] = imp_num.transform(X_te[num_cols])

X_tr[cat_cols] = imp_cat.fit_transform(X_tr[cat_cols])
X_te[cat_cols] = imp_cat.transform(X_te[cat_cols])
```

**Perché `fit_transform` su train e `transform` su test?** Il primo *apprende* i parametri (mediana, moda...) dal training set; il secondo li *applica* senza guardare il test set → niente data leakage.

### 1.6 Encoding delle categoriche

Per semplicità si usa **un solo `OrdinalEncoder`** su tutte le colonne categoriche, senza separare nominali da ordinali (non serve creare due encoder distinti):

```python
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

enc = OrdinalEncoder()
X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
X_te[cat_cols] = enc.transform(X_te[cat_cols])

# encoding del TARGET se categoriale testuale (es. multiclasse)
le = LabelEncoder()
y_tr = le.fit_transform(y_tr)
y_te = le.transform(y_te)
```

*(esiste anche `OneHotEncoder` per le variabili puramente nominali, ma nella pratica del corso si usa quasi sempre il solo `OrdinalEncoder` su tutte le categoriche, per semplicità)*

### 1.7 Standardizzazione

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)
```

$$z_{ij} = \frac{x_{ij}-\mu_i}{\sigma_i}$$

⚠️ **In regressione: si scalano SOLO le feature, mai il target `y`.**

### 1.8 Analisi di correlazione

**Visualizzazione rapida** (utile come primo sguardo d'insieme, soprattutto nei compiti d'esame):

```python
matrice_corr = X_tr.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(matrice_corr, cmap='coolwarm', center=0, annot=True)
```

**a) Feature poco correlate col target** (solo se supervisionato — **NON** in clustering, dove il target non esiste concettualmente):

```python
soglia = 0.1
corr_target = X_tr.corrwith(y_tr).abs().sort_values(ascending=False)
col_to_remove = corr_target[corr_target < soglia].index

X_tr = X_tr.drop(columns=col_to_remove)
X_te = X_te.drop(columns=col_to_remove)
```

**b) Multicollinearità tra feature** (sempre, anche in clustering):

```python
soglia = 0.9

coppie = matrice_corr.unstack().reset_index()
coppie.columns = ['var1', 'var2', 'corr']
coppie = coppie[coppie['var1'] != coppie['var2']]

cols_to_remove = coppie[coppie['corr'].abs() >= soglia]
cols_to_remove
```

Per le coppie ridondanti individuate, capita spesso che il professore chieda di **sostituirle con la loro combinazione lineare** (media per riga) invece di rimuoverne semplicemente una:

```python
X_tr['X23'] = X_tr[['X2', 'X3']].mean(axis=1)
X_tr['X1011'] = X_tr[['X10', 'X11']].mean(axis=1)
X_tr['X1415'] = X_tr[['X14', 'X15']].mean(axis=1)
X_tr = X_tr.drop(columns=['X2', 'X3', 'X10', 'X11', 'X14', 'X15'])

X_te['X23'] = X_te[['X2', 'X3']].mean(axis=1)
X_te['X1011'] = X_te[['X10', 'X11']].mean(axis=1)
X_te['X1415'] = X_te[['X14', 'X15']].mean(axis=1)
X_te = X_te.drop(columns=['X2', 'X3', 'X10', 'X11', 'X14', 'X15'])

X_tr.head()
```

In alternativa, se le variabili multicollineari sono molte, si può applicare **PCA** per ridurre la dimensionalità:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_tr_pca = pca.fit_transform(X_tr)
X_te_pca = pca.transform(X_te)
```
PCA riduce la dimensionalità ma **fa perdere interpretabilità**.

### 1.9 Feature Selection

| Metodo | Tipo | Come funziona | Quando |
|---|---|---|---|
| `SelectKBest` | filter | test statistico (`f_classif`/ANOVA per classificazione, `f_regression` per regressione) tra ogni feature e il target | semplice, veloce, non considera interazioni tra feature |
| `SelectFromModel` | embedded | usa l'importanza/coefficienti di un modello già addestrato (Lasso, RandomForest, SVC lineare...) | cattura interazioni, soglia automatica (es. `threshold='median'`) |

```python
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel

skb = SelectKBest(score_func=f_classif, k=5) # k=5 prende solo le 5 piu importanti k='all' tutte
X_tr_skb = skb.fit_transform(X_tr, y_tr)
X_te_skb = skb.transform(X_te)

# embedded, esempio con RandomForest
from sklearn.ensemble import RandomForestClassifier
sfm = SelectFromModel(RandomForestClassifier(random_state=42), threshold='median') #threshold='median' posso modificarlo 
X_tr_sfm = sfm.fit_transform(X_tr, y_tr)
X_te_sfm = sfm.transform(X_te)
```

**Metodi utili di `SelectKBest`** (comodi per ispezionare cosa è stato selezionato senza ricostruire le colonne a mano):
```python
skb.get_support()        # array booleano: True per le feature selezionate
skb.feature_names_in_    # nomi di TUTTE le feature in input (nell'ordine originale)
skb.pvalues_              # p-value del test per ciascuna feature
```

Modelli utilizzabili in `SelectFromModel`:
- **Classificazione**: `LogisticRegression(penalty='l1', solver='liblinear')`, `LinearSVC(penalty='l1', dual=False)`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `HistGradientBoostingClassifier`
- **Regressione**: `Lasso`/`LassoCV` (con `GridSearchCV` su `alpha`), `SVR(kernel='linear')`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `Ridge`

---

## 2. Capacità del modello / Bias-Variance trade-off

### Concetto chiave
La **capacità** di un modello è la sua flessibilità ad adattarsi a funzioni complesse (es. il grado $p$ di un polinomio). All'aumentare di $p$:
- ↓ **bias** (il modello si adatta meglio ai dati di training)
- ↑ **varianza** (rischio di **overfitting**, scarsa generalizzazione)

La **capacità ottima** è quella che minimizza il $\text{MSE}_{test}$ (non quello di train, che decresce sempre).

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

### Outlier detection (utile prima di fittare modelli sensibili agli outlier)

**Local Outlier Factor (LOF)** — basato su densità *locale* relativa ai vicini:

```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=32, contamination=0.25)   # k ≈ sqrt(n) come euristica k è n_neighbors
labels = lof.fit_predict(X)     # 1 = inlier, -1 = outlier
inlier = labels!= -1
X = X[inlier] #cosi temgo solo quelli con 1, ossia quelli che non sono outlier
```

**Kernel Density Estimation** — alternativa basata su densità di probabilità:

```python
from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(X) # ora c'è fit 
score = kde.score_samples(X)
soglia = pd.Series(score).quantile(0.2)
inlier_mask = score > soglia
X = X[inlier_mask]
```

### GridSearchCV per il tuning (esempio con KernelRidge)
# per ottimizzare gli iperparametri del modello, trovare i migliori.
# ( La si può usare anche nella feature selection ma ha poco senso pratico perche in quel caso mi interessa solamente trovare le variabili piu importanti )

```python
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

# creo un dizionario dei valori da usare (o li da lui o li scelgo io)
param = {'alpha': [0.001, 0.1, 1], 'gamma': [0.001, 0.01, 0.05]} # alpha e gamma perche in questo caso sto usando KernelRidge. per lasso per esempio solo alpha
gs = GridSearchCV(KernelRidge(kernel='rbf'), param_grid=param, scoring='r2', cv=5, n_jobs=-1) # per la regressione scoring='r2', per la classificazione  scoring='accuracy'
# se dovesse chiedere più di uno, potei scrivere: scoring=['r2', 'mean_squared_error']
gs.fit(X_tr, y_tr)      # alleno il modello migliore sui miei dati di train
gs.best_params_         # restituisce i migliori iperparametri 
gs.best_score_          # trovo la migliore metrica, che in questo caso è r2
gs.best_estimator_      # il miglior modello
mod= gs.best_estimator_ # do un nome al miglior modello ottenuto
y_pred = mod.predict(X_te) # usando il modello miglire fa la predizione sull'X test
```

**Tecnica utile — ricerca a due passaggi:** se il `best_params_` cade sul **bordo** della griglia (es. il valore più alto/basso testato), è segno che l'intervallo era troppo stretto. Conviene rilanciare la `GridSearchCV` con una seconda griglia più ampia/fine **centrata attorno al valore trovato**, per affinare la stima:

```python
# 1° passaggio: griglia larga, esplorativa
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
grid = GridSearchCV(Lasso(max_iter=10000), param_grid=param_grid, scoring='r2', cv=5)
grid.fit(X_tr, y_tr)   # best alpha = 10.0 → è il bordo, ristringo la zoomata

# 2° passaggio: griglia raffinata attorno al valore trovato
param_grid = {'alpha': [5.0, 10.0, 20.0, 25.0, 30.0, 100.0, 1000.0]}
grid = GridSearchCV(Lasso(max_iter=10000), param_grid=param_grid, scoring='r2', cv=5)
grid.fit(X_tr, y_tr)
```

---

## 3. Clustering (apprendimento non supervisionato)

⚠️ **Differenze chiave rispetto al supervisionato:**
- **non si fa** analisi di correlazione feature-target (il target, se esiste, è solo per validazione esterna, non per fare model selection)
- concettualmente **non avrebbe senso** fare train/test split (il clustering scopre struttura, non impara una regola predittiva) — ma se la consegna lo richiede, si fa lo stesso
- la **standardizzazione** resta importante, così come PCA / feature selection per ridurre dimensionalità

### Algoritmi principali

| Algoritmo | Come funziona | Pro | Contro |
|---|---|---|---|
| **K-Means** | minimizza distanza euclidea² tra punti e centroidi | semplice, veloce, ottimo con cluster sferici e ben separati | richiede K a priori, assume cluster sferici, sensibile a outlier |
| **DBSCAN** | basato su densità: punti *core/border/outlier* in base a `MinPts` e raggio `Eps` | non richiede K, trova forme arbitrarie, isola outlier (label -1) | soffre la curse of dimensionality, fatica con densità variabili |
| **HDBSCAN** | estensione gerarchica di DBSCAN, gestisce densità variabili | non richiede `Eps`, solo `MinPts` | più lento |
| **Gaussian Mixture (GMM)** | mix di gaussiane stimate con EM, **clustering probabilistico** (assegna probabilità, non etichetta netta) | cattura cluster ellissoidali, più flessibile di K-Means | richiede K a priori, assume gaussianità |

### Scelta di K — Elbow method (K-Means)

```python
from sklearn.cluster import KMeans
inertia = []
for k in [2,3,4,5,6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertia.append(km.inertia_)
# plottare inertia vs k, cercare il "gomito"
```

### Metriche di valutazione

| Metrica | Tipo | Range | Uso |
|---|---|---|---|
| `silhouette_score` | interna (no ground truth) | [-1, 1] | quanto i punti sono coerenti col proprio cluster vs gli altri; serve ≥2 cluster |
| `adjusted_rand_score` (ARI) | esterna (richiede ground truth) | ~[-1, 1], 1=perfetto, 0=casuale | confronta etichette predette con quelle vere |
| **Purity** | esterna | [0, 1] | $\sum_k \max_j |cluster_k \cap classe_j| / n$ |
| **Accuracy** (via matrice di contingenza) | esterna | [0, 1] | solo se # cluster = # classi (matrice quadrata): max tra diagonale e diagonale invertita |

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

score = silhouette_score(X, labels)
ari = adjusted_rand_score(y_true, labels)

cm = contingency_matrix(y_true, labels)
purity = np.sum(np.amax(cm, axis=0)) / cm.sum()
# accuracy SOLO se cm è quadrata 2x2 (caso binario):
accuracy = max(cm[0,0]+cm[1,1], cm[0,1]+cm[1,0]) / cm.sum()
```

### Tuning iperparametri per DBSCAN (grid search manuale)

```python
from sklearn.cluster import DBSCAN
best_score, best_params = -1, None
for min_pts in [5, 10, 20]:
    for eps in [0.5, 1.5, 2.0]:
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
        if len(set(labels)) > 2:   # serve almeno 2 cluster veri (oltre al rumore)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_params = score, (eps, min_pts)
```

**Pattern alternativo — risultati in un DataFrame:** se si dispone già del ground truth (`y_train`) e si vuole usare direttamente l'**ARI** come criterio (invece del silhouette), è comodo salvare ogni combinazione testata in un DataFrame e poi ordinare:

```python
mod_selection = pd.DataFrame(columns=["MinPts", "eps", "ari"])
id_row = 0
for t in [5, 10, 20]:
    for e in [0.5, 1.0, 2.0]:
        id_row += 1
        db = DBSCAN(min_samples=t, eps=e)
        y_pred = db.fit_predict(X_tr)
        ari = adjusted_rand_score(labels_true=y_tr, labels_pred=y_pred)
        mod_selection.loc[id_row] = [t, e, ari]

mod_selection.sort_values(by="ari", ascending=False)        # tutte le combinazioni, ordinate
best_row = mod_selection.loc[mod_selection["ari"].idxmax()]  # solo la migliore, miglior modello in base all'ari
```
Utile perché permette di **ispezionare tutta la griglia** (non solo il best), comodo per commentare nel compito perché certe combinazioni funzionano meglio di altre.
```python
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=2, weights_init=[1/2] * 2,random_state=42) # weights_init=[1/2] * 2 --> 2 distribuzioni ognuna con peso di 1/2
# 3. Addestriamo e generiamo direttamente le etichette per il Punto 4
y_pred_gm = gm.fit_predict(X_tr_kbest)
gm.weights_ # pesi ottenuti
```

### Come interpretare quale algoritmo "vince"
- **K-Means vince** → classi reali formano cluster sferici/compatti/ben separati
- **DBSCAN vince** → forme strane/non lineari, o molto rumore che DBSCAN isola bene
- **GMM vince** → classi sovrapposte parzialmente, forme ellittiche/dimensioni diverse

---

## 4. Classificazione (binaria e multiclasse)

### Modelli principali (con `GridSearchCV` per il tuning)

```python
from sklearn.model_selection import GridSearchCV

# Random Forest
from sklearn.ensemble import RandomForestClassifier
param_grid = {'criterion': ['gini', 'log_loss'], 'min_samples_split': [2,5,10], 'max_features': ['sqrt', 5]}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.01,0.1,1,10,100], 'penalty': ['l1','l2']}   # solver='liblinear'

# SVC
from sklearn.svm import SVC
param_grid = {'kernel': ['linear','rbf'], 'C': [0.1,1,10], 'gamma': ['scale','auto']}

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
param_grid = {'n_estimators': [50,100,200], 'learning_rate': [0.01,0.1,1]}

grid = GridSearchCV(estimator=modello, param_grid=param_grid, cv=5, n_jobs=-1) # n_jobs=-1 usa tutti i core disponibili, posso cambiarli 
grid.fit(X_tr, y_tr)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_te)
```

### Metriche — Classificazione binaria

```python
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, RocCurveDisplay

print(classification_report(y_te, y_pred))             # precision, recall, f1, support per classe
accuracy_score(y_te, y_pred)                            # accuracy come singolo numero

RocCurveDisplay.from_estimator(best_model, X_te, y_te)   # ROC + AUC, versione grafica

# Versione "manuale" per ottenere il valore numerico dell'AUC (utile se serve solo il numero)
y_prob = best_model.predict_proba(X_te)[:, 1]            # probabilità della classe positiva
roc_auc_score(y_te, y_prob)

# Se mi dovesse chiedere di confrontare le metriche del train e del test devo :
# prendere le metrice del test  usando la funzione accuracy_score(y_te, y_pred) 
# prendere le metrice del train usando mod.best_score_ 
```

### Metriche — Classificazione multiclasse

Per il target serve `LabelEncoder` (se testuale). La ROC va calcolata in modalità **One-vs-Rest**:

```python
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_auc_score

y_onehot = LabelBinarizer().fit_transform(y_te)
y_score = best_model.predict_proba(X_te)

# questo codice for solo se chiede la rappresentazione delle curve ROC per ogni classe
for i, classe in enumerate(best_model.classes_):
    RocCurveDisplay.from_predictions(y_onehot[:, i], y_score[:, i], name=f"ROC {classe}")

# questi due a seguire DEVO RIPORTARLI 
macro_auc = roc_auc_score(y_te, y_score, multi_class="ovr", average="macro")
micro_auc = roc_auc_score(y_te, y_score, multi_class="ovr", average="micro")
auc_medio = roc_auc_score(y_te, y_score, multi_class="ovr", average= None)
```

---

## 5. Regressione

### Modelli principali

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Random Forest Regressor (tuning tipico)
param_grid = {'criterion': ['squared_error','absolute_error'], 'min_samples_split': [2,5,10], 'max_features': ['sqrt',5]}

# Lasso / Ridge
param_grid = {'alpha': [0.001,0.01,1,10], 'tol': [0.001,0.01,0.05]}

grid = GridSearchCV(estimator=modello, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_tr, y_tr)
```

⚠️ **Nota su alpha (Lasso/Ridge):** $\alpha$ basso → penalità debole, simile a regressione lineare standard; $\alpha$ alto → penalità aggressiva, coefficienti vanno a 0 (modello inutile/sottoadattato).

### Metriche

```python
from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_te, y_pred)                          # quota di varianza spiegata, ideale vicino a 1
mse = mean_squared_error(y_te, y_pred)           # errore quadratico medio
rmse = np.sqrt(mse)                              # stessa unità di misura del target, più interpretabile dell'MSE

print(f'RMSE: {rmse:.2f}  R2: {r2_score(y_te, y_pred):.5f}')
```

---

## 6. Cheat-sheet finale: quale metrica/modello scegliere

| Task | Target | Modelli tipici | Metriche |
|---|---|---|---|
| Classificazione binaria | discreto, 2 classi | LogisticRegression, RandomForestClassifier, SVC, AdaBoost | classification_report, accuracy_score, ROC/AUC (display o `roc_auc_score`) |
| Classificazione multiclasse | discreto, >2 classi | gli stessi + `predict_proba` | classification_report, ROC one-vs-rest, macro/micro AUC |
| Regressione | continuo | LinearRegression, Ridge, Lasso, RandomForestRegressor, SVR | R², MSE, RMSE, residual plot |
| Clustering | nessuno (non supervisionato) | K-Means, DBSCAN, HDBSCAN, GMM | silhouette (interna), ARI/purity/accuracy (esterna, se hai ground truth) |

### Promemoria trasversali da non dimenticare mai
1. **Split prima di tutto** il resto (imputazione, encoding, scaling, feature selection) → `fit_transform` su train, `transform` su test.
2. **Stratify=y** nello split se è un problema di classificazione (bilanciamento classi); in regressione, se serve, si può stratificare su una **feature categorica** importante.
3. **Mai correlazione col target in clustering** (non supervisionato).
4. **Mai scalare il target** in regressione.
5. Ordinal vs OneHot in base a se la categorica ha un **ordine logico** o no.
6. `set_config(transform_output='pandas')` per avere sempre DataFrame invece di array numpy puri.
7. Rimuovi anche le colonne **identificative/ad alta cardinalità** (ID, nomi), non solo quelle sparse.
8. Se in `GridSearchCV` il `best_params_` cade sul **bordo** della griglia, allarga/raffina la ricerca con un secondo passaggio.

---

## Nota — confronto con i 3 compiti d'esame svolti

I tre file caricati (*Classificazione binaria*, *Classificazione multiclasse*, *Regressione*, tutti con la parte di rete neurale tagliata) **confermano esattamente** la pipeline descritta sopra: sparsità/cardinalità → split → imputazione → encoding → standardizzazione → correlazione → feature selection → modello + metriche. Le uniche differenze sono di stile (sintassi alternative equivalenti, qualche passaggio omesso quando non richiesto dal compito) — tutte le varianti utili sono state integrate nelle sezioni sopra.
