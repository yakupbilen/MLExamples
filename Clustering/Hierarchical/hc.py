import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../../Classification/Iris.csv")

x = df.iloc[:,1:5]

dendrogram = sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()

ac = AgglomerativeClustering(n_clusters=3)
pred = ac.fit_predict(x)

for i in range(0,len(x.columns)):
    for j in range(i+1,len(x.columns)):
        data = pd.concat((df[x.columns[i]],df[x.columns[j]]),axis=1)
        plt.scatter(data.iloc[pred == 0, 0], data.iloc[pred == 0, 1],label=0)
        plt.scatter(data.iloc[pred == 1, 0], data.iloc[pred == 1, 1],label=1)
        plt.scatter(data.iloc[pred == 2, 0], data.iloc[pred == 2, 1],label=2)
        plt.xlabel(x.columns[i])
        plt.ylabel(x.columns[j])
        plt.legend()
        plt.show()

