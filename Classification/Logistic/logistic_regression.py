import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Preprocessing.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler

df = pd.read_csv("Social_Network_Ads.csv")
df.drop(columns=["User ID"],inplace=True)
print("Null values")
print(df.isna().sum())

df = OneHotEncoder().fit_transform(df,["Gender"])
# Correlation Matrix
print(df.corr(method="pearson"))

x = df.iloc[:,0:3].values  # => ["Gender","Age","EstimatedSalary"]
y = df.iloc[:,3:4].values  # => ["Purchased"]

# Classification with Unscaled Data

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=42)

logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train.ravel())
y_pred=logr.predict(x_test)

print(f"\n\nModel 1(Unscaled): \n{confusion_matrix(y_test,y_pred)}")

# Classification with MinMaxScaled Data
minmax_sc = MinMaxScaler(feature_range=(0,1))
x_minmax = minmax_sc.fit_transform(x) # Gender column doesn't change!! (value/1 = value)

x_train, x_test, y_train, y_test = train_test_split(x_minmax,y,train_size=0.7,random_state=42)

logr = LogisticRegression(random_state=3)
logr.fit(x_train,y_train.ravel())
y_pred=logr.predict(x_test)

print(f"\nModel 2(MinMaxScaled Data): \n{confusion_matrix(y_test,y_pred)}")

# Classification with StandartScaled Data
gender = df.iloc[:,0:1].values  # The gender column is not scaled because it consists of 0s and 1s.
x = df.iloc[:,1:3].values

standart_sc = StandardScaler()
x_sc = standart_sc.fit_transform(x)
x_sc = np.concatenate((gender,x_sc),axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_sc,y,train_size=0.7,random_state=42)

logr = LogisticRegression(random_state=3)
logr.fit(x_train,y_train.ravel())
y_pred=logr.predict(x_test)

print(f"\nModel 3(StandartScaled Data): \n{confusion_matrix(y_test,y_pred)}")
