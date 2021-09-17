import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../Knn/diabetes.csv")

print("Null values")
print(df.isna().sum())

print("\nCorrelation of Columns")
print(df.corr(method="pearson"))
x = df.iloc[:, 0:8].values
y = df.iloc[:, 8:9].values

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=44)

svm = SVC(kernel="linear")
svm.fit(x_train,y_train.ravel())

pred = svm.predict(x_test)

print(classification_report(y_test,pred,digits=6))
print(confusion_matrix(y_test,pred))
"""
 random_state = 3

 linear:
 [[116  17]
 [ 44  54]]
 
 poly:(degree=3)
 [[121  12]
 [ 54  44]]
 
 poly:(degree=4)
 [[121  12]
 [ 58  40]]

 rbf:
 [[120  13]
 [ 57  41]]

"""