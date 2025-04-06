import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load Data
data = pd.read_csv("indiv-project/water_potability.csv")
data = data.dropna()

# Feature and Target Split
x = data.drop(columns=["Potability"])
y = data["Potability"]

# Function for Class Mapping
def map_potability(value):
    return "Unsafe" if value == 0 else "Safe"

# Plot Original Distribution
plt.figure(figsize=(6,4))
sb.countplot(x=y.map(map_potability), palette=["red", "blue"])
plt.title("Distribution of classes (Potability)")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.show()

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to Balance the Dataset
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Plot Balanced Data Distribution
plt.figure(figsize=(6,4))
sb.countplot(x=y_train.map(map_potability), palette=["blue", "red"])
plt.title("Distribution of classes (Potability) after balancing")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.show()

# Initialize Classifiers with Default Parameters
rf_model = RandomForestClassifier(random_state=42)
et_model = ExtraTreesClassifier(random_state=42)

# Train Models
rf_model.fit(x_train, y_train)
et_model.fit(x_train, y_train)

# Make Predictions
rf_predictions = rf_model.predict(x_test)
et_predictions = et_model.predict(x_test)

# Evaluate Random Forest
print("=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_predictions))

# Evaluate Extra Trees
print("=== Extra Trees Classification Report ===")
print(classification_report(y_test, et_predictions))

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix - Random Forest
sb.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt="d", cmap="coolwarm",
           xticklabels=["Unsafe", "Safe"], yticklabels=["Unsafe", "Safe"], ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_ylabel("Predicted")
axes[0].set_xlabel("Actual")

conf_matrix1=confusion_matrix(y_test,rf_predictions)
print(conf_matrix1)

# Confusion Matrix - Extra Trees
sb.heatmap(confusion_matrix(y_test, et_predictions), annot=True, fmt="d", cmap="coolwarm",
           xticklabels=["Unsafe", "Safe"], yticklabels=["Unsafe", "Safe"], ax=axes[1])
axes[1].set_title("Extra Trees Confusion Matrix")
axes[1].set_ylabel("Predicted")
axes[1].set_xlabel("Actual")
conf_matrix=confusion_matrix(y_test,et_predictions)
print(conf_matrix)

plt.show()