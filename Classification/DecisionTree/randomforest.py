import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../Knn/diabetes.csv")
print(len(df))


x = df.iloc[:,0:8].values
y = df.iloc[:,8:9].values

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=0)

classifier = RandomForestClassifier(n_estimators=10,random_state=0)

classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)

print(confusion_matrix(y_test,pred))

"""
[[141  16]
 [ 39  35]]
 """