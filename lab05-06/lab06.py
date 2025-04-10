import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\Canada_Completed.xlsx")

plt.figure(figsize = (10,8))
corr_matrix = df.corr()
print(corr_matrix.round(2))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix of Canada's features")
plt.tight_layout()
plt.show()

x = df[['industry']]
y = df['gdp_growth']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("Coefficient of Determination: ", r2)

plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Industry Growth (%)')
plt.ylabel('GDP Growth (%)')
plt.title('GDP Growth vs Industry Growth Regression')
plt.legend()
plt.grid(True)
plt.show()
