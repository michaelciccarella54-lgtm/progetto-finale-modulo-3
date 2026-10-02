# PROGETTO FINALE - MODULO 3
# Pipeline completa di Machine Learning per la regressione
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
# 1. DATA EXPLORATION & PREPROCESSING
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
Y = pd.DataFrame(diabetes.target, columns=["target"])
print("=== PRIME 5 RIGHE DELLE FEATURE ===")
print(X.head())
print("\n=== PRIME 5 RIGHE DEL TARGET ===")
print(Y.head())
y = Y.values.ravel()
print("\n=== ANALISI DESCRITTIVA ===")
print(X.describe())
print("\n=== TIPI DI DATO ===")
print(X.dtypes)
print("\n=== VALORI MANCANTI ===")
print(X.isnull().sum())
X.hist(figsize=(12, 8))
plt.suptitle("Distribuzione delle Feature")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
X.boxplot()
plt.title("Boxplot delle Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
pd.plotting.scatter_matrix(X, figsize=(12, 8), diagonal='hist')
plt.suptitle("Scatter Matrix delle Feature")
plt.tight_layout()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n=== TRAIN/TEST SPLIT ===")
print("Dimensioni X_train:", X_train.shape)
print("Dimensioni X_test:", X_test.shape)
print("Dimensioni y_train:", y_train.shape)
print("Dimensioni y_test:", y_test.shape)
# 2. CONFRONTO E SELEZIONE DEI MODELLI
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Ridge" : Ridge(),
    "Lasso" : Lasso(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR()
}
print("\n=== MODELLI CREATI ===")
for nome in models:
    print(nome)
kFold = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n=== K-FOLD CROSS VALIDATION ===")
for nome, modello in models.items():
    scores = cross_val_score(modello, X_train, y_train, cv=kFold, scoring= "neg_mean_squared_error")
    print (f"{nome} : NMSE medio = {scores.mean() : .2f}")
# 3. HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV
print ("\n=== HYPERPARAMETER TUNING ===")
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=kFold, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)
print("Miglior parametro:", grid_search.best_params_)
print("Miglior NMSE:", grid_search.best_score_)
# 3.2 VALUTAZIONE FINALE
from sklearn.metrics import mean_squared_error, r2_score
print("\n=== VALUTAZIONE FINALE ===")
modello_finale = grid_search.best_estimator_
y_pred = modello_finale.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE sul test set: {mse:.2f}")
print(f"R^2 sul test set: {r2:.2f}")
# 4. ANALISI DELLE PERFORMANCE
from sklearn.model_selection import learning_curve
print("\n=== ANALISI DELLE PERFORMANCE ===")
train_sizes, train_scores, test_scores = learning_curve(modello_finale, X_train, y_train, cv=kFold, scoring="neg_mean_squared_error", train_sizes=np.linspace(0.1, 1.0, 10))
train_mse = -train_scores.mean(axis=1)
validation_mse = -test_scores.mean(axis=1)
print("MSE di training:", train_mse[-1])
print("MSE di validazione:", validation_mse[-1])
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mse, label="Training MSE", marker='o')
plt.plot(train_sizes, validation_mse, label="Validation MSE", marker='o')
plt.xlabel("Numero di campioni di training")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve - Ridge Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
gap = validation_mse[-1] - train_mse[-1]
print("\n=== BIAS / VARIANCE ===")
if gap > 1000:
    print("Possibile overfitting: alta varianza")
elif validation_mse[-1] > 4000:
    print("Possibile underfitting: bias elevato")
else:
    print("Il modello mostra un buon equilibrio tra bias e varianza")
# 5. VISUALIZZAZIONE CON DATA REDUCTION
# 5.1 PCA - RIDUZIONE A 2 DIMENSIONI
from sklearn.decomposition import PCA
print("\n=== PCA - DATA REDUCTION ===")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("Dimensioni originali:", X_train.shape)
print("Dimensioni dopo PCA:", X_train_pca.shape)
print ("Varianza spiegata PC1 e PC2:", pca.explained_variance_ratio_)
# 5.2 PIANO DI REGRESSIONE NELLO SPAZIO PCA
modello_pca = Ridge(alpha=grid_search.best_params_["alpha"])
modello_pca.fit(X_train_pca, y_train)
print("Modello Ridge addestrato sulle 2 componenti PCA")
pc1 = np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 30)
pc2 = np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 30)
PC1, PC2 = np.meshgrid(pc1, pc2)
griglia = np.c_[PC1.ravel(), PC2.ravel()]
predizioni_griglia = modello_pca.predict(griglia)
Z = predizioni_griglia.reshape(PC1.shape)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
ax.plot_surface(PC1, PC2, Z, alpha=0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Target")
ax.set_title("Piano di regressione Ridge nello spazio PCA")
plt.show()
# CONFRONTO VALORI REALI E VALORI PREDETTI
print("\n=== CONFRONTO VALORI REALI E PREDETTI ===")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Valori Reali")
plt.ylabel("Valori Predetti")
plt.title("Valori Reali vs Valori Predetti")
plt.grid(True)
plt.tight_layout()
plt.show()

# CONCLUSIONI FINALI
print("\n=== CONCLUSIONI FINALI ===")
print("Modello selezionato: Ridge Regression")
print("Miglior parametro alpha:", grid_search.best_params_["alpha"])
print(f"MSE sul test set: {mse:.2f}")
print(f"R^2 sul test set: {r2:.2f}")
print("\nProgetto completato con successo!")