# coding:utf-8


"""
    ML Challenge 02 - Predict bikes bookings
    Build pipeline with Scikit-learn
    Linear Regression
"""


###  PACKAGES  ###
import pickle
from colorama import Fore

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


### PATHS & CONFIG ###
DATA = "./data/bikes-data.csv"
CORR  = "./outputs/02-corr.png"
MODEL = "./outputs/02-model.pkl"
TEST = 0.2


### GET DATA ###
print(f"Load data from {DATA}")
data = pd.read_csv(DATA)
print(data.head())

### EXPLORE DATA ###
print(data.info())
n_samples = len(data)
print("Total datas:", n_samples)

# Correlation matrix
continuous_columns = list(data.columns)[1:-1]
corr = data[continuous_columns].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cbar=True, fmt=".0%")
plt.title("Correlation matrix")
plt.savefig(CORR)

### SAMPLE DATA ###
X = data.drop(["count"], axis=1)
y = data["count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST, random_state=42)

### PREPROCESS DATA ###
scaler = ColumnTransformer([("scaler", StandardScaler(), continuous_columns)])

### MODEL ###
lr = LinearRegression(n_jobs=-1)

### BUILD PIPELINE ###
pipeline = Pipeline([
    ("scaler", scaler),
    ("lr", lr)
])

### TRAIN MODEL ###
pipeline.fit(X_train, y_train)

### EVALUATE MODEL ###
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

### SAVE MODEL ###
with open(MODEL, "wb") as f:
    print(f"Save model at {MODEL}")
    pickle.dump(pipeline, f)

### TEST SAVED MODEL ###
with open(MODEL, "rb") as f:
    saved_pipeline = pickle.load(f)

try:
    sample = X_test.sample(1)
    assert pipeline.predict(sample)[0] == saved_pipeline.predict(sample)[0]
    print(Fore.GREEN + "Model works properly ↑↑↑")
except Exception as e:
    print(Fore.RED + "Something turns wrong ↓↓↓", e)
