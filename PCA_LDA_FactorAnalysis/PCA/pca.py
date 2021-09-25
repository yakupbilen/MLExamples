import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
from sklearn.datasets import load_wine

data = load_wine()

df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= data['feature_names'] + ['target'])

print("Null values")
print(df.isna().sum())

x = df.iloc[:,0:13]
y = df.iloc[:,13:14]


best_comp = 0
best_acc = 0

for i in range(2,len(x.columns)):
    pca = PCA(n_components=i)
    x_pca = pca.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, train_size=0.7, random_state=0)

    svm = SVC(kernel="linear", random_state=0)
    svm.fit(x_train, y_train.values.ravel())
    svm_pred = svm.predict(x_test)
    acc = accuracy_score(y_test,svm_pred)
    print(i,"Components Accuracy : ",acc)

    if acc>best_acc:
        best_acc = acc
        best_comp = i


pca = PCA(n_components=best_comp)
x_pca = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, train_size=0.7, random_state=0)
svm = SVC(kernel="linear", random_state=0)
svm.fit(x_train, y_train.values.ravel())
svm_pred = svm.predict(x_test)

print(pca.n_components_,"Components")
print(confusion_matrix(y_test,svm_pred))
print("Accuracy : ",accuracy_score(y_test,svm_pred))

