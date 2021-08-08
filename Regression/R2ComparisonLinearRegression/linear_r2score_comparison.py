import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt


df = pd.read_csv("advertising.csv")
r2_scores = []


for column in df:
    if column != "Sales":
        X = df[[column]]
        y = df[["Sales"]]

        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=33)

        lr = LinearRegression()
        lr.fit(x_train,y_train)
        predict = lr.predict(x_test)
        r2_scores.append([r2_score(y_test,predict),column])

model_base = max(r2_scores)
print(50*"*")
print(f"Selected independent feature is '{model_base[1]}'")
print(f"Feature to be predicted is 'Sales'")
print(50*"*")

X = df[[model_base[1]]]
Y = df[["Sales"]]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=19)
X_test = x_test.sort_index()  # for plotting
Y_test = y_test.sort_index()

lr = LinearRegression()
lr.fit(x_train, y_train)
predict = lr.predict(X_test)
r2 = r2_score(Y_test, predict)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_test.shape[1]))
print(f"{model_base[1]} --> Sales r2 score : {r2}")
print(f"{model_base[1]} --> Sales Adjusted r2 score : {adj_r2}")
print(f"{model_base[1]} --> Sales Mean Absolute Error: {mean_absolute_error(Y_test, predict)}")
print(f"{model_base[1]} --> Sales Mean Squared Error: {mean_squared_error(Y_test, predict)}\n")

plt.scatter(X_test, Y_test)
plt.plot(X_test, predict)
plt.savefig(model_base[1]+"-Sales")
