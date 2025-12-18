#!/usr/bin/env python
# coding: utf-8

# Importation des librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Chargement du dataset
df = pd.read_csv(r'C:\Users\lenovo\Downloads\DATASET_Combiner.csv')

# Garder uniquement les colonnes numériques
df_numeric = df.select_dtypes(include=['number'])

# Encodage de la colonne 'Position'
le = LabelEncoder()
position_encoded = le.fit_transform(df['Position'])
df_numeric['Position_Code'] = position_encoded

# Affichage de la matrice de corrélation
print(df_numeric.corr())

# Sélection des features et de la cible
features = [
    'Matches_Played',
    'Assists',
    'Goals_Assists',
    'Non_Penalty_Goals',
    'Penalty_Goals',
    'Penalties_Attempted',
    'Expected_Goals',
    'Non_Penalty_xG',
    'Expected_Assisted_Goals',
    'Non_Penalty_xG_plus_xAG',
    'Progressive_Receptions'
]
target = "Goals"

X = df[features]
y = df[target]

# Pipeline de prétraitement
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), features)
])

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du modèle
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Entraînement du modèle
model_gb.fit(X_train, y_train)

# Prédictions
y_pred = model_gb.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Gradient Boosting Regressor")
print(f"---------------------------")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

# Sauvegarde du modèle
joblib.dump(model_gb, 'C:\\Users\\lenovo\\Desktop\\web mining\\ambilance\\gradient_boosting_model.pkl')
print("Modèle enregistré avec succès au format .pkl")
