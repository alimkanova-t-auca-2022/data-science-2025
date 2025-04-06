import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pycaret.classification import *
from imblearn.over_sampling import SMOTE

data = pd.read_csv("indiv-project/water_potability.csv")

data = data.dropna()

#classfication = setup(data, target="Potability", session_id=786)
#compare_models()

x = data.drop(columns=["Potability"])
y = data["Potability"]

def map_potability(value):
    return "Unsafe" if value==0 else "Safe"

plt.figure(figsize=(6,4))
sb.countplot(x=y.map(map_potability), palette=["red","blue"])
plt.title("Distribution of classes (Potability)")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

plt.figure(figsize=(6,4))
sb.countplot(x=y_train.map(map_potability), palette=["blue","red"])
plt.title("Distribution of classes (Potability) after balancing")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.show()


param_tuning = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ['gini', 'entropy'],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
                           param_grid=param_tuning,
                           cv=5,
                           n_jobs=3,
                           verbose=2,
                           scoring="accuracy")

grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Best parameters: ", best_params)

model = RandomForestClassifier(n_estimators=best_params["n_estimators"],
                               max_depth=best_params["max_depth"],
                               min_samples_split=best_params["min_samples_split"],
                               min_samples_leaf=best_params["min_samples_leaf"],
                               criterion=best_params["criterion"],
                               class_weight="balanced",
                               random_state=42)

model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

print("Classification Report: \n", classification_report(y_test, y_prediction))

plt.figure(figsize=(5,4))
sb.heatmap(confusion_matrix(y_test, y_prediction), annot=True, fmt="d", cmap="coolwarm",
           xticklabels=["Unsafe", "Safe"], yticklabels=["Unsafe", "Safe"])
plt.title("Matrix of Errors")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
conf_matrix = confusion_matrix(y_test, y_prediction)
print("Матрица ошибок:")
print(conf_matrix)