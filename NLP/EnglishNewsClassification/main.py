import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,SnowballStemmer
warnings.filterwarnings("ignore")

models = {}
stop_words = set(stopwords.words('english'))
df = pd.read_csv("news-article-categories.csv")
df['body'] = df['body'].apply(str)
df["body"] = df["body"].apply(lambda a: a.lower())
df["body"] = df["body"].apply(lambda a: [word for word in word_tokenize(a) if word not in stop_words and word.isalnum()])

stemmer = PorterStemmer()
stemmer2 = SnowballStemmer(language="english")
df_snow = df.copy()
df_porter = df.copy()

df_porter["body"] = df_porter["body"].apply(lambda a: [stemmer.stem(word) for word in a])
df_snow["body"] = df_snow["body"].apply(lambda a: [stemmer2.stem(word) for word in a])

df_porter["body"] = df_porter["body"].apply(lambda a: " ".join([word for word in a]))
df["body"] = df["body"].apply(lambda a: " ".join([word for word in a]))
df_snow["body"] = df_snow["body"].apply(lambda a: " ".join([word for word in a]))


# No stemming
x = df["body"].tolist()
y = df["category"].values


x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.75, random_state=15)

pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("svc",LinearSVC())])
pipe.fit(x_train,y_train)
pred = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred): ["No Stemming -> LinearSVC",pred]})

# Porter Stemmer
x = df_porter["body"].tolist()
y = df_porter["category"].values

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.75, random_state=15)

pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("svc",LinearSVC())])
pipe.fit(x_train,y_train)
pred_porter = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_porter): ["PorterStemmer -> LinearSVC",pred_porter]})


# Snowball Stemmer
x = df_snow["body"].tolist()
y = df_snow["category"].values

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=15)

pipe = Pipeline([("tfidf",TfidfVectorizer()),
                 ("svc",LinearSVC())])
pipe.fit(x_train,y_train)
pred_snow = pipe.predict(x_test)
models.update({accuracy_score(y_test, pred_snow): ["SnowballStemmer -> LinearSVC",pred_snow]})

choice = max(models)
print("Best Model is between No Stemming -> LinearSVC, PorterStemmer -> LinearSVC, SnowballStemmer -> LinearSVC")
print(f"{models[choice][0]} \n Accuracy : {choice}")
print("Confusion Matrix : \n", confusion_matrix(y_test, models[choice][1]))
print("Classification Report : \n", classification_report(y_test, models[choice][1]))
