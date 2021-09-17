import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../../Classification/Iris.csv")

x = df.iloc[:,1:5]

scores = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    scores.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=2,init='k-means++',random_state=0)
pred = kmeans.fit_predict(x)

labels = np.unique(pred)

for i in range(0,len(x.columns)):
    for j in range(i+1,len(x.columns)):
        data = pd.concat((df[x.columns[i]],df[x.columns[j]]),axis=1)
        plt.scatter(data.iloc[pred == 0, 0], data.iloc[pred == 0, 1],label=0)
        plt.scatter(data.iloc[pred == 1, 0], data.iloc[pred == 1, 1],label=1)
        plt.xlabel(x.columns[i])
        plt.ylabel(x.columns[j])
        plt.legend()
        plt.show()