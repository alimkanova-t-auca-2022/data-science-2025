import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("indiv-project/water_potability.csv")

# Drop missing values
data = data.dropna()

# Define features (X) and target (y)
X = data.drop(columns=["Potability"])  # Assuming 'Potability' is the target variable
y = data["Potability"]

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(x_train, y_train)

# Get feature importance
feature_importance_rf = rf_model.feature_importances_

# Create a DataFrame for visualization
features = X.columns  # Use original column names
importance_df_rf = pd.DataFrame({"Feature": features, "Importance": feature_importance_rf})

# Sort values for better visualization
importance_df_rf = importance_df_rf.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df_rf["Feature"], importance_df_rf["Importance"], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()  # Display highest importance at the top
plt.show()
