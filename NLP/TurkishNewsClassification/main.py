import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from TurkishStemmer import TurkishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("7allV03.csv")
encoder = LabelEncoder()
df["category"] = encoder.fit_transform(df["category"])

stemmer = TurkishStemmer()
df["text"] = df["text"].apply(lambda a: [stemmer.stem(word) for word in a.split()])
df["text"] = df["text"].apply(lambda a: " ".join([word for word in a]))


x = df["text"].tolist()
y = df["category"].values

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=15)


models = {}

# Multinomial Naive Bayes
pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("mnb",MultinomialNB())])

pipe.fit(x_train,y_train)
pred_mnb_tf = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_mnb_tf): ["Bag Of Words ->Multinomial NB",pred_mnb_tf]})


pipe = Pipeline([("cv",CountVectorizer()),
                 ("mnb",MultinomialNB())])

pipe.fit(x_train,y_train)
pred_mnb_cv = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_mnb_cv): ["TfIdf -> Multinomial NB",pred_mnb_cv]})

# KNN
pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("knn",KNeighborsClassifier(n_neighbors=7))])

pipe.fit(x_train,y_train)
pred_knn_tf = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_knn_tf): ["Tf Idf ->Knn",pred_knn_tf]})


pipe = Pipeline([('cv', CountVectorizer()),
                     ('knn', KNeighborsClassifier(n_neighbors=7)),
                     ])
pipe.fit(x_train,y_train)
pred_knn_cv = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_knn_cv): ["Bag Of Words ->Knn",pred_knn_cv]})

# Support Vector Machine


pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("svc",LinearSVC())])

pipe.fit(x_train,y_train)
pred_svc_tf = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_svc_tf): ["Tf Idf ->LinearSVC",pred_svc_tf]})


pipe = Pipeline([('cv', CountVectorizer()),
                     ('svc', LinearSVC()),
                     ])
pipe.fit(x_train,y_train)
pred_svc_cv = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_svc_cv): ["Bag Of Words -> LinearSVC",pred_svc_cv]})


# Decision Tree

pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("tree",DecisionTreeClassifier())])

pipe.fit(x_train,y_train)
pred_tree_tf = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_tree_tf): ["Tf Idf ->Decision Tree",pred_tree_tf]})


pipe = Pipeline([('cv', CountVectorizer()),
                     ('tree', DecisionTreeClassifier()),
                     ])
pipe.fit(x_train,y_train)
pred_tree_cv = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_tree_cv): ["Bag Of Words -> Decision Tree",pred_tree_cv]})

# Random Forest

pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("rand_tree",RandomForestClassifier(n_estimators=100))])

pipe.fit(x_train,y_train)
pred_randtree_tf = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_randtree_tf): ["Tf Idf -> Random Forest",pred_randtree_tf]})


pipe = Pipeline([('cv', CountVectorizer()),
                     ('rand_tree', RandomForestClassifier(n_estimators=100)),
                     ])
pipe.fit(x_train,y_train)
pred_randtree_cv = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_randtree_cv): ["Bag Of Words -> Random Forest",pred_randtree_cv]})

print("Accuracy : ")
for key, value in models.items():
    print(value[0],"->",key)


choice = max(models)
print("\n\nThe best model is ",models[choice][0])
print("Confusion Matrix : \n", confusion_matrix(y_test,models[choice][1]))
print("Accuracy : ", choice)
print("Classification Report : \n",classification_report(y_test, models[choice][1]))