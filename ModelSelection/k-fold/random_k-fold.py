import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("../../Classification/Knn/diabetes.csv")

x = df.iloc[:, 0:8]
y = df.iloc[:, 8:9].values

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=2)

model = GaussianNB()

scores = cross_val_score(model,x,y.ravel(),cv=cv)
print("Mean accuracy of repeated k-fold model with 5 repeats, 10 splits : ",scores.mean())
print("Standart Deviation : ",scores.std())
