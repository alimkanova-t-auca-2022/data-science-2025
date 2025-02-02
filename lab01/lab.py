import pandas as pd
import math
import csv

file = open("lab01/ages.txt", "w")
file.write("12.7, 15.3, 16.2, 10.5, 12, 14.0, 13.1, 16.3")
file.close()

n = int( input("n: "))
file = open("lab01/ages.txt", "r")
ages = file.read()
file.close()

ages = ages.split(",")
ages = [float(age) for age in ages[:n]]

for i in range(len(ages)):
    ages[i] = math.floor(ages[i])

scores = [
    ["Name", "Score"],
    ["Andrew", 88.3],
    ["Ben", 92.6],
    ["Carol", 89.7]
]

file = open("lab01/scores.csv", "w", newline="")
csv_wr = csv.writer(file)
csv_wr.writerows(scores)
file.close()

file = open("lab01/scores.csv", "r")
csv_re = csv.reader(file)

next(csv_re) #skipping the header

dic = []

for row in csv_re:
    dic.append({"Name": row[0], "Score": float(row[1])})
file.close()

for i in range(len(dic)):
    dic[i] = {"Name": dic[i]["Name"], "Age": ages[i], "Score": dic[i]["Score"]}

data_frame = pd.DataFrame(dic)
print("DataFrame:")
print(data_frame)

max_age = data_frame['Age'].max()
print(f"Maximum age: {max_age}")

row = pd.DataFrame([{"Name": "Daniel", "Age":10, "Score": 78}])
data_frame = pd.concat([data_frame, row], ignore_index=True)

p = float(input("Enter deduction p: "))
for i in range(len(data_frame)):
    if data_frame.loc[i, 'Name'] == 'Ben':
        data_frame.loc[i, 'Score'] -=p

for i in range(len(data_frame)):
    data_frame.loc[i, 'Score'] = math.ceil(data_frame.loc[i,'Score']*2)/2

mean_score = data_frame['Score'].mean()
median_score = data_frame['Score'].median()
print("\nUpdated DataFrame:")
print(data_frame)
print(f"Mean Score: {mean_score}")
print(f"Median Score: {median_score}")

dataframe_filter = data_frame[(data_frame['Age'] >= 12) & (data_frame['Age'] <= 15)]
print("\nFiltered DataFrame:")
print(dataframe_filter)

mean_filter = dataframe_filter['Score'].mean()
median_filter = dataframe_filter['Score'].median()

print(f"Mean Score in Filtered DataFrame: {mean_filter}")
print(f"Median Score: {median_filter}")