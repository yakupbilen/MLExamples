import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score
from Preprocessing.outlier import delete_outliers_qr

# From correlation matrix

""" 
    Coefficient of attributes 
Pregnancies = 3
Glucose = 6
BloodPressure = 1
SkinThickness = 1
Insulin = 2
BMI = 4
DiabetesPedigreeFunction = 2
Age = 3

"""

df = pd.read_csv("diabetes.csv")


print("Null values")
print(df.isna().sum())


print("\nCorrelation of Columns")
print(df.corr(method="pearson"))
df = delete_outliers_qr(df,["Outcome"])
x = df.iloc[:, 0:8]
y = df.iloc[:, 8:9]


for column in x.columns:
    max = x[column].max()
    min = x[column].min()
    x[column]=x[column].apply(lambda value : ((value-min)/(max-min)))

x["Pregnancies"] = x["Pregnancies"].apply(lambda value: value*3)
x["Glucose"] = x["Glucose"].apply(lambda value: value*6)
x["Insulin"] = x["Insulin"].apply(lambda value: value*2)
x["BMI"] = x["BMI"].apply(lambda value: value*4)
x["DiabetesPedigreeFunction"] = x["DiabetesPedigreeFunction"].apply(lambda value: value*2)
x["Age"] = x["Age"].apply(lambda value: value*3)


"""
minmax_sc = MinMaxScaler(feature_range=(0,1))
x_minmax = minmax_sc.fit_transform(x) # Gender column doesn't change!! (value/1 = value)
"""

"""
standart_sc = StandardScaler()
x_sc = standart_sc.fit_transform(x)
"""



x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=22)

knn = KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn.fit(x_train,y_train.values.ravel())
pred = knn.predict(x_test)
cm = confusion_matrix(y_test,pred)
print("\nConfusion Matrix : ")
print(cm)
print("Accuracy : ",accuracy_score(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred))

"""
 random state = 42
 weighted
[[119  32]
 [ 36  44]]
 minmax(always range 0-1)
[[119  32]
 [ 39  41]]
 standartscaled
 [[123  28]
 [ 39  41]]
 
 random state = 2
 weighted
 [[135  20]
 [ 36  40]]
 minmax(always range 0-1)
 [[135  20]
 [ 40  36]]
 standartscaled
 [[133  22]
 [ 37  39]]
"""