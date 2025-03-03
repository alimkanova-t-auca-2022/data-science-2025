import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

data_frame = pd.read_csv("lab03\PL_Players.csv", encoding='latin-1')
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

x = data_frame[["Tackles", "Tackles+Interceptions", "Blocked the ball", "Passes blocked", "Interceptions", "Tackles Won","Successful press","number of times opponent put under pressure", "% of successful press", "Tackles won in defensive 3rd","Tackles won in midfield", "Tackles won in attacking 3rd"]]

y = data_frame["Position"] 

encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, stratify=y, random_state =42)

n = int(input("Enter n: "))
k_nearest_n = KNeighborsClassifier(n_neighbors = n)
k_nearest_n.fit(x_train, y_train)
prediction = k_nearest_n.predict(x_test)

print("Accuracy:", accuracy_score(y_test, prediction))
print("Classification Report:\n", classification_report(y_test,prediction, zero_division=1))