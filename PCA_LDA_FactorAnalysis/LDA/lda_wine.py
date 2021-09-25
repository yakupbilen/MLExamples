import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
from sklearn.datasets import load_wine

data = load_wine()

df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= data['feature_names'] + ['target'])


x = df.iloc[:,0:13]
y = df.iloc[:,13:14]

lda = LinearDiscriminantAnalysis(n_components=2)
x = lda.fit_transform(x,y.values.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

svm = SVC(kernel="linear", random_state=0)
svm.fit(x_train, y_train.values.ravel())
svm_pred = svm.predict(x_test)
print("With 2 components(LDA) Accuracy is ")
print(accuracy_score(y_test,svm_pred))
print("*"*50)
print(confusion_matrix(y_test,svm_pred))