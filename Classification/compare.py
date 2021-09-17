import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("Knn/diabetes.csv")
print(df.head(5))


x = df.iloc[:,0:8].values
y = df.iloc[:,8:9].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Logistic
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,train_size=0.7,random_state=0)
logr = LogisticRegression()
logr.fit(x_train,y_train.ravel())
logr_pred = logr.predict(x_test)

# Knn
knn = KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn.fit(x_train,y_train.ravel())
knn_pred = knn.predict(x_test)

# Svm
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=0)
svm = SVC(kernel="linear")
svm.fit(x_train,y_train.ravel())
svm_pred = svm.predict(x_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_pred = nb.predict(x_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)

# Random Forest
rdt = RandomForestClassifier(n_estimators=15,random_state=0)
rdt.fit(x_train,y_train)
rdt_pred = rdt.predict(x_test)


print(f'Logistic Regression\n{confusion_matrix(y_test,logr_pred)}')
print(f'Accuracy : {accuracy_score(y_test,logr_pred)}\nPrecision : {precision_score(y_test,logr_pred)}\n'
      f'Recall : {recall_score(y_test,logr_pred)}\nF1 Score : {f1_score(y_test,logr_pred)}\n\n')
print(f'Knn Classifier\n{confusion_matrix(y_test,knn_pred)}')
print(f'Accuracy : {accuracy_score(y_test,knn_pred)}\nPrecision : {precision_score(y_test,knn_pred)}\n'
      f'Recall : {recall_score(y_test,knn_pred)}\nF1 Score : {f1_score(y_test,knn_pred)}\n\n')
print(f'SVM Classifier\n{confusion_matrix(y_test,svm_pred)}')
print(f'Accuracy : {accuracy_score(y_test,svm_pred)}\nPrecision : {precision_score(y_test,svm_pred)}\n'
      f'Recall : {recall_score(y_test,svm_pred)}\nF1 Score : {f1_score(y_test,svm_pred)}\n\n')
print(f'Naive Bayes Classifier\n{confusion_matrix(y_test,nb_pred)}')
print(f'Accuracy : {accuracy_score(y_test,nb_pred)}\nPrecision : {precision_score(y_test,nb_pred)}\n'
      f'Recall : {recall_score(y_test,nb_pred)}\nF1 Score : {f1_score(y_test,nb_pred)}\n\n')
print(f'Decision Tree Classifier\n{confusion_matrix(y_test,dt_pred)}')
print(f'Accuracy : {accuracy_score(y_test,dt_pred)}\nPrecision : {precision_score(y_test,dt_pred)}\n'
      f'Recall : {recall_score(y_test,dt_pred)}\nF1 Score : {f1_score(y_test,dt_pred)}\n\n')
print(f'Random Forest Classifier\n{confusion_matrix(y_test,rdt_pred)}')
print(f'Accuracy : {accuracy_score(y_test,rdt_pred)}\nPrecision : {precision_score(y_test,rdt_pred)}\n'
      f'Recall : {recall_score(y_test,rdt_pred)}\nF1 Score : {f1_score(y_test,rdt_pred)}')


