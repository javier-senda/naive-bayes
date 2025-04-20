import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## Paso 1: Carga del conjunto de datos
total_data= pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv")
total_data.to_csv("../data/raw/total_data.csv", index = False)

## Paso 2: Estudio de variables y su contenido

### Eliminar columna irrelevante

total_data.drop("package_name", axis=1, inplace=True)

### Eliminar espacios y convertir texto a minúsculas

total_data["review"] = total_data["review"].str.strip().str.lower()

### División en train y test

X = total_data["review"]
y = total_data["polarity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_excel("../data/processed/X_train.xlsx", index = False)
X_test.to_excel("../data/processed/X_test.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

### Transformar el texto en matriz de recuento de palabras

vec_model = CountVectorizer(stop_words = "english")
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()

## Paso 3: Construye un naive bayes

### MultinomialNB: Elegimos esta implementación porque los datos son discretos

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

### GaussianNB y BernoulliNB

for model2 in [GaussianNB(), BernoulliNB()]:
    model2.fit(X_train,y_train)
    y_pred_train2 = model2.predict(X_train)
    y_pred_test2 = model2.predict(X_test)
    print(model2)
    print(f"Train: {accuracy_score(y_train, y_pred_train2)}")
    print(f"Test: {accuracy_score(y_test, y_pred_test2)}")

#### La implementación de MultinomialNB era la adecuada, ofrece mejor accuracy en test que las otras 2

## Paso 4: Optimiza el modelo anterior

param_grid = {
    "alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "fit_prior": [True, False],
}

grid = GridSearchCV(model, param_grid, scoring = "accuracy", cv=5)
grid.fit(X_train, y_train)

final_model = grid.best_estimator_
final_model.fit(X_train, y_train)

y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

## Paso 5: Guardar el modelo

with open("../models/MultinombialNB_best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

## Paso 6: Explora otras alternativas

### Regresión logística

log_reg_model = LogisticRegression(random_state=42)

log_reg_model.fit(X_train, y_train)

y_pred_train = log_reg_model.predict(X_train)
y_pred_test = log_reg_model.predict(X_test)

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

### Boosting

boosting_model = XGBClassifier(random_state = 42)

boosting_model.fit(X_train, y_train)

y_pred_train = boosting_model.predict(X_train)
y_pred_test = boosting_model.predict(X_test)

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

### Random Forest

random_forest_model = RandomForestClassifier(random_state = 42)

random_forest_model.fit(X_train, y_train)

y_pred_train = random_forest_model.predict(X_train)
y_pred_test = random_forest_model.predict(X_test)

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

### Decision tree

decision_tree_model = DecisionTreeClassifier(random_state = 42)

decision_tree_model.fit(X_train, y_train)

y_pred_train = decision_tree_model.predict(X_train)
y_pred_test = decision_tree_model.predict(X_test)

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

### Conclusión

#### Ningún modelo mejora a la implementación MultinomialNB 