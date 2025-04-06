import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# Load Data
data = pd.read_csv("indiv-project/water_potability.csv")
data = data.dropna()

# Feature and Target Split
x = data.drop(columns=["Potability"])
y = data["Potability"]

# Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Define Hyperparameter Grid for Randomized Search
param_dist = {
    "n_estimators": np.arange(100, 500, 50),
    "max_depth": [10, 20, 30, 40, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "criterion": ["gini", "entropy"],
}

# Initialize Extra Trees Classifier
et_model = ExtraTreesClassifier(random_state=42)

# Perform Randomized Search
random_search = RandomizedSearchCV(et_model, param_distributions=param_dist, cv=5, n_jobs=-1, verbose=2, scoring="accuracy", n_iter=20, random_state=42)
random_search.fit(x_train, y_train)

# Best Parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Train Best Model
optimized_et = ExtraTreesClassifier(**best_params, random_state=42)
optimized_et.fit(x_train, y_train)

# Make Predictions
et_predictions = optimized_et.predict(x_test)

# Evaluate Model
print("=== Optimized Extra Trees Classification Report ===")
print(classification_report(y_test, et_predictions))

# Confusion Matrix
plt.figure(figsize=(5,4))
sb.heatmap(confusion_matrix(y_test, et_predictions), annot=True, fmt="d", cmap="coolwarm",
           xticklabels=["Unsafe", "Safe"], yticklabels=["Unsafe", "Safe"])
plt.title("Optimized Extra Trees Classifier - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()