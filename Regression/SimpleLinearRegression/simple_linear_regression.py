import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("Linear Regression.csv")

x = df[["X"]]
y = df[["Y"]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


X_test = x_test.sort_index()
Y_test = y_test.sort_index()

lr = LinearRegression()
lr.fit(x_train,y_train)
predict = lr.predict(X_test)
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,predict,color="black")
print(type(X_test))
plt.show()