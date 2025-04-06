import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_frame = pd.read_csv("lab04\PL_Players.csv", encoding='latin-1')
data_frame.dropna(inplace=True)
data_frame = data_frame.rename(columns={"Pos" : "Position",
                           "90s" : "Total minutes played divided by 90",
                           "Tkl" : "Tackles",
                           "TklW" : "Tackles Won",
                           "Def 3rd.1" : "Tackles won in defensive 3rd",
                           "Mid 3rd.1" : "Tackles won in midfield",
                           "Att 3rd.1" : "Tackles won in attacking 3rd",
                           "Press" : "number of times opponent put under pressure",
                           "Succ" : "Successful press",
                           "%" : "% of successful press",
                           "Def 3rd" : "Press in defense",
                           "Mid 3rd" : "Press in midfield",
                           "Att 3rd" : "Press in attack",
                           "Blocks" : "Blocked the ball",
                           "Pass" : "Passes blocked",
                           "Int" : "Interceptions",
                           "Tkl+Int" :"Tackles+Interceptions",
                           "Clr" : "Clearances",
                           "Err" : "Errors leading to goal"})
# Normalize the 'Position' column to ensure uniform naming
data_frame["Position"] = data_frame["Position"].str.strip().str.upper()  # Convert to uppercase & remove spaces

# Optional: If there are known variations of position names, map them to a standard format
position_mapping = {
    "FW": "FORWARD",
    "MF": "MIDFIELDER",
    "DF": "DEFENDER",
    "GK": "GOALKEEPER",
}

# Apply mapping to standardize position names
data_frame["Position"] = data_frame["Position"].replace(position_mapping)


print("Position distribuition: ")
print(data_frame["Position"].value_counts())
features = ["Tackles", "Tackles+Interceptions", "Blocked the ball", "Passes blocked", "Interceptions", "Tackles Won","Successful press","number of times opponent put under pressure", "% of successful press", "Tackles won in defensive 3rd","Tackles won in midfield", "Tackles won in attacking 3rd"]
train_set, test_set = train_test_split(data_frame, test_size=0.2, random_state=42, stratify=data_frame["Position"] ) 
train_set, valid_set = train_test_split(train_set, test_size=0.25, random_state=42, stratify=train_set["Position"] )

x_train, y_train = train_set[features], train_set["Position"] 
x_valid, y_valid = valid_set[features], valid_set["Position"] 
x_test, y_test = test_set[features], test_set["Position"] 


dec_tree_model = DecisionTreeClassifier(random_state=42)
scores_tree = cross_val_score(dec_tree_model, x_train, y_train, cv=5)
dec_tree_model.fit(x_train, y_train)
y_prediction = dec_tree_model.predict(x_test)

print("Decision Tree accuracy: ", accuracy_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction, zero_division=1))

best_accuracy = 0
best_depth = 0
best_min_sample_leaf = 0
for depth in range(1,15):
    for min_samples_leaf in range(1,11):
        dec_tree_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples_leaf,random_state=42)
        scores = cross_val_score(dec_tree_model, x_train, y_train, cv=5)
        average_score = scores.mean()

        if average_score > best_accuracy:
            best_accuracy = average_score
            best_depth = depth
            best_min_sample_leaf = min_samples_leaf

print(f"Best Decision Tree depth: {best_depth}, Min Sample Leaf: {best_min_sample_leaf}")

best_dec_tree_model = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_min_sample_leaf, random_state=42)
best_dec_tree_model.fit(x_train, y_train)
y_prediction = best_dec_tree_model.predict(x_test)

print("Optimized Decision Tree accuracy: ", accuracy_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction, zero_division=1))

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_prediction_knn = knn.predict(x_test)
print("KNN Accuracy: ", accuracy_score(y_test, y_prediction_knn))
print(classification_report(y_test, y_prediction_knn, zero_division=1))

fw_mf = data_frame[data_frame["Position"].isin(["FW","MF"])]
x_logistic = fw_mf[features] 
y_logistic = fw_mf['Position']
fw_label ="FW"
y_logistic = (y_logistic==fw_label).astype(int)

logistic_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
logistic_reg.fit(x_logistic, y_logistic)
y_scores = logistic_reg.predict_proba(x_logistic)[:,1]

false_pos_r, true_pos_r, threshold = roc_curve(y_logistic, y_scores)
roc_auc = auc(false_pos_r, true_pos_r)

pyplot.figure()
pyplot.plot(false_pos_r, true_pos_r, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
pyplot.plot([0,1], [0,1], color='gray', linestyle='--')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC Curve for FW vs. MF Classification')
pyplot.legend()
pyplot.show()

random_threshold = 0.5
y_pred_random = (y_scores >= random_threshold).astype(int)
print(f"Accuracy at threshold {random_threshold}: {accuracy_score(y_logistic, y_pred_random):.2f}")