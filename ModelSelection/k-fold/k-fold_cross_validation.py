import pandas as pd
from sklearn.model_selection import cross_val_score,KFold
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("../../Classification/Knn/diabetes.csv")

x = df.iloc[:, 0:8]
y = df.iloc[:, 8:9].values

cv = KFold(n_splits=10,random_state=2,shuffle=True)

model = GaussianNB()

scores = cross_val_score(model,x,y.ravel(),cv=cv)
print("Mean of 10 splits model accuracy : ",scores.mean())
print("Standart Deviation : ",scores.std())
