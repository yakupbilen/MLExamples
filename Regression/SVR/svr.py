from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

data = pd.read_csv("../R2ComparisonLinearRegression/advertising.csv")

r2_scores = []
for col in data:
    if col != "Sales":
        scaler1 = StandardScaler()
        X_scale = scaler1.fit_transform(data[[col]])
        scaler2 = StandardScaler()
        y_scale = scaler2.fit_transform(data[["Sales"]])

        X_train,X_test,y_train,y_test = train_test_split(X_scale,y_scale,test_size=0.3,random_state=53)

        svr = SVR()
        svr.fit(X_train,y_train.ravel())
        y_pred = svr.predict(X_test)
        r2_scores.append([r2_score(y_test,y_pred), col])

model = max(r2_scores)

scaler1 = StandardScaler()
X_scale = scaler1.fit_transform(data[[model[1]]])
scaler2 = StandardScaler()
y_scale = scaler2.fit_transform(data[["Sales"]])

X_train,X_test,y_train,y_test = train_test_split(X_scale,y_scale,test_size=0.3,random_state=53)

svr = SVR()
svr.fit(X_train,y_train.ravel())
y_pred = svr.predict(X_test)
r2 = r2_score(y_test,y_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_scale.shape[1]))
print(f"SVR R2 Score : {r2}")
print(f"SVR Adjusted R2 Score : {adj_r2}")



