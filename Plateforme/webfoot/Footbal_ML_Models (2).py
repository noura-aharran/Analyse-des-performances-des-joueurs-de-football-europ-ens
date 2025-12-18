#!/usr/bin/env python
# coding: utf-8

# À partir de ce code, nous souhaitons trouver un modèle performant permettant de prédire le nombre de buts qu'un joueur peut marquer durant un match.

# ## Importation des données

# In[102]:

# Importation des librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# In[103]:

# Chargement du dataset
df = pd.read_csv(r"C:\Users\lenovo\Downloads\DATASET_Combiner.csv")

# Maintenant, nous voulons conserver uniquement les colonnes numériques (int, float) afin de pouvoir calculer la corrélation entre ces colonnes par la suite.

# In[104]:

df_numeric = df.select_dtypes(include=['number'])

# Nous allons également encoder la colonne position pour pouvoir inclure sa corrélation dans l’analyse. 

# In[105]:

# Créer un encodeur
le = LabelEncoder()

# Encoder la colonne Position (du DataFrame original)
position_encoded = le.fit_transform(df['Position'])
df_numeric['Position_Code'] = position_encoded

# In[106]:

# Calcul de la corrélation
print(df_numeric.corr())

# Sélection des features ayant forte corrélation (> 0.5) avec la cible

# In[107]:

features = [
    'Matches_Played',
    'Assists',
    'Goals_Assists',
    'Non_Penalty_Goals',
    'Penalty_Goals',
    'Penalties_Attempted',
    'Expected_Goals',
    'Non_Penalty_xG',
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

# ## 1.1. Random Forest Regressor

# In[108]:

model1 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# In[109]:

# Entraînement du modèle
model1.fit(X_train, y_train)

# Prédictions
y_pred = model1.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Regressor")
print(f"---------------------")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

# ## 1.2. XGBRegressor

# In[110]:

model2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# In[111]:

model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"XGBRegressor")
print(f"---------------------")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

# ## 1.3. Gradient Boosting Regressor

# In[113]:

model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# In[114]:

model_gb.fit(X_train, y_train)

y_pred = model_gb.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Gradient Boosting Regressor")
print(f"---------------------")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

# ## 1.4. Linear Regression

# In[115]:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

y_pred = model_lr.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression")
print(f"-----------------")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Enregistrement du meilleur modèle (supposons Gradient Boosting ici)

joblib.dump(model_gb, 'C:\\Users\\lenovo\\Desktop\\web mining\\ambilance\\gradient_boosting_model.pkl')

print("Modèle enregistré avec succès au format .pkl")
