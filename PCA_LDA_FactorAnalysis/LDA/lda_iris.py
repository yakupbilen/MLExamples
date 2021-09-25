import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv("../../Classification/Iris.csv")

print("Null values")
print(df.isna().sum())

x = df.iloc[:,1:5]
y = df.iloc[:,5:6]

y = LabelEncoder().fit_transform(y.values.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())
pred = knn.predict(x_test)
print("Accuracy : ",accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))


lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit_transform(x,y.ravel())

x_train, x_test, y_train, y_test = train_test_split(x_lda, y, train_size=0.7, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())
pred = knn.predict(x_test)
print("\n2 components Accuracy : ",accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))



