import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../Knn/diabetes.csv")
print(df.head(5))


x = df.iloc[:,0:8].values
y = df.iloc[:,8:9].values

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=0)

classifier = DecisionTreeClassifier(random_state=2)

classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)

print(confusion_matrix(y_test,pred))