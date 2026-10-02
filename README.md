# Progetto Finale Modulo 3 - Machine Learning

Progetto finale del Modulo 3 dedicato alla costruzione e valutazione di modelli di regressione utilizzando il dataset Diabetes di Scikit-learn.

## Obiettivo

L'obiettivo del progetto è sviluppare una pipeline completa di Machine Learning per un problema di regressione, confrontando diversi algoritmi e analizzandone le prestazioni.

## 1. Data Exploration & Preprocessing

- Caricamento del dataset Diabetes
- Analisi descrittiva dei dati
- Controllo dei valori mancanti
- Analisi delle distribuzioni tramite istogrammi
- Analisi degli outlier tramite boxplot
- Analisi delle relazioni tra feature tramite Scatter Matrix
- Suddivisione dei dati in Training Set e Test Set

## 2. Confronto e selezione dei modelli

Sono stati confrontati sei algoritmi di regressione:

- Linear Regression
- Decision Tree Regressor
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors Regressor
- Support Vector Regression (SVR)

Il confronto è stato effettuato tramite K-Fold Cross Validation utilizzando il Negative Mean Squared Error come metrica.

## 3. Hyperparameter Tuning

È stata utilizzata GridSearchCV per ottimizzare l'iperparametro `alpha` del modello Ridge Regression.

Il modello ottimizzato è stato successivamente valutato sul Test Set utilizzando:

- Mean Squared Error (MSE)
- R² Score

## 4. Analisi delle Performance

Sono state generate le Learning Curves confrontando l'errore sul Training Set e sul Validation Set.

I risultati vengono utilizzati per analizzare:

- Bias
- Variance
- Possibile overfitting
- Possibile underfitting

## 5. Visualizzazione con Data Reduction

È stata applicata la Principal Component Analysis (PCA) per ridurre le feature originali a due componenti principali:

- PC1
- PC2

Il modello Ridge è stato successivamente rappresentato nello spazio PCA tramite un piano di regressione tridimensionale.

## Visualizzazioni

Il progetto genera diverse visualizzazioni:

- Istogrammi delle feature
- Boxplot
- Scatter Matrix
- Learning Curve
- Piano di regressione nello spazio PCA
- Confronto tra valori reali e valori predetti

## Tecnologie utilizzate

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Esecuzione

Installare le dipendenze:

```bash
pip install -r requirements.txt