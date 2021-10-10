import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")

data= load_breast_cancer()

df = pd.DataFrame(np.c_[data['data'], data['target']],
                  columns= np.append(data['feature_names'], ['target']))

x = df.iloc[:, :30]
y = df.iloc[:, 30:]

"""print("Null values")
print(df.isna().sum())"""

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

svc = SVC(kernel="linear", random_state=0)
svc.fit(x_train, y_train.values.ravel())
pred = svc.predict(x_test)
print("Accuracy : ",accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


kmo_all, kmo_model=calculate_kmo(x)
print("\n\nA value of KMO less than 0.6 is considered insufficient for Factor Analysis")
print("KMO value : ", kmo_model)


fa = FactorAnalyzer(6)
x = fa.fit_transform(x)
ev, v = fa.get_eigenvalues()
"""print("EigenValues : ",ev)
print("Factor Scores : (Recommended to choose greater than 0,6-0,7)",fa.loadings_)"""

plt.plot(range(0, 30), ev)
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

svc = SVC(kernel="linear", random_state=0)
svc.fit(x_train, y_train.values.ravel())
pred = svc.predict(x_test)
print("\n","*"*10,"After Factor Analsis","*"*10)
print("Accuracy : ",accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
