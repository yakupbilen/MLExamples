import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Preprocessing.preprocessing import backward_elimination

data = pd.read_csv("../MultiLinearRegression/USA_Housing.csv")

del data["Address"]
y = data[["Price"]]
del data["Price"]
data = backward_elimination(data, y)
y = y.values

x_train, x_test, y_train, y_test = train_test_split(data, y, train_size=0.3, random_state=1)

regressor = RandomForestRegressor(n_estimators=12, random_state=0)

regressor.fit(x_train, y_train.ravel())
y_pred = regressor.predict(x_test)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-x_test.shape[1]))
print(f"Random Forest Regressor R2 Score : {r2}")
print(f"Random Forest Regressor Adjusted R2 Score : {adj_r2}")
print(f"Random Forest Regressor Mean Squared Error : {mean_squared_error(y_test, y_pred)}")

