import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from Preprocessing.preprocessing import backward_elimination
from sklearn.model_selection import train_test_split


df = pd.read_csv("../MultiLinearRegression/USA_Housing.csv")

df.drop(columns=["Address"],axis=1,inplace=True)
print(f"{20*'-'}Correlation Matrix{20*'-'}")
print(df.corr())

y = df[["Price"]]
df.drop(columns=["Price"],axis=1,inplace=True)
df = backward_elimination(df, y)
X = df



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)


# Linear Regression
lr =LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
r2 = r2_score(y_test,lr_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_test.shape[1]))
print(f"\n\nLinear R2 Score : {r2}")
print(f"Linear Adjusted R2 Score : {adj_r2}")


# Polynomial Regression(degree = 2)
# In this data set, I think that a degree of 2 is better than a degree of 3
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
poly2 = PolynomialFeatures(degree=2)
X_test_poly = poly2.fit_transform(X_test)

lr2 = LinearRegression()
lr2.fit(X_train_poly,y_train)
poly_pred = lr2.predict(X_test_poly)
r2 = r2_score(y_test, poly_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_test.shape[1]))
print(f"\n\nPoly(degree = 2) R2 Score : {r2}")
print(f"Poly(degree = 2) Adjusted R2 Score : {adj_r2}")


# Polynomial Regression (degree = 3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
poly2 = PolynomialFeatures(degree=3)
X_test_poly = poly2.fit_transform(X_test)

lr2 = LinearRegression()
lr2.fit(X_train_poly,y_train)
poly_pred = lr2.predict(X_test_poly)
r2 = r2_score(y_test, poly_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_test.shape[1]))
print(f"\n\nPoly(degree = 3) R2 Score : {r2}")
print(f"Poly(degree = 3) Adjusted R2 Score : {adj_r2}")


# Support Vector Regression
scaler1 = StandardScaler()
y_scale = scaler1.fit_transform(y)
scaler2 = StandardScaler()
X_scale = scaler2.fit_transform(X)
X_s_train,X_s_test,y_s_train,y_s_test = train_test_split(X_scale,y_scale,test_size=0.3,random_state=5)
svr = SVR(kernel="rbf")
svr.fit(X_s_train,y_s_train.ravel())
svr_pred = svr.predict(X_s_test)
r2 = r2_score(y_s_test, svr_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_s_test.shape[1]))
print(f"\n\nSVR(kernel = 'rbf') R2 Score : {r2}")
print(f"SVR(kernel = 'rbf') Adjusted R2 Score : {adj_r2}")


# Random Forest Regression
random = RandomForestRegressor(n_estimators=12,random_state=1)
random.fit(X_train,y_train.values.ravel())
rand_pred = random.predict(X_test)
r2 = r2_score(y_test, rand_pred)
adj_r2 = 1-(1-r2)*((len(y_test)-1)/(len(y_test)-1-X_s_test.shape[1]))
print(f"\n\nRandom Forest(12 estimators) R2 Score : {r2}")
print(f"Random Forest(12 estimators) Adjusted R2 Score : {adj_r2}")



