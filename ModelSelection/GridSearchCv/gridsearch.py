import pandas as pd
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../../Classification/Knn/diabetes.csv")

x = df.iloc[:, 0:8]
y = df.iloc[:, 8:9].values

cv = KFold(n_splits=10,random_state=2,shuffle=True)

model = RandomForestClassifier()
params = [{"n_estimators":range(5,20),"criterion":["gini","entropy"]}]
gs = GridSearchCV(estimator=model,param_grid=params,cv=cv)
gs = gs.fit(x,y.ravel())

print("Best parameters : ",gs.best_params_)
print("Best accuracy score : ",gs.best_score_)
