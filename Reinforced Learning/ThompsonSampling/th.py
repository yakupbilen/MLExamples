import random

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Ads_CTR_Optimisation.csv")
print(df.head(5))

len_col = len(df.columns)
len_row = len(df)

zeros = [0]*len_col
ones = [0]*len_col
total_rewards = 0

selected_ad = []
for i in range(0,len_row):
    ad = 0
    max_th = 0
    for j in range(0,len_col):
        randbeta = random.betavariate(ones[j]+1,zeros[j]+1)
        if randbeta>max_th:
            max_th = randbeta
            ad = j
    selected_ad.append(ad)
    if df.values[i,ad] == 1: # 0 -> 10000
        ones[ad] +=1
        total_rewards += 1
    else:
        zeros[ad] +=1

print(f"Total Reward : {total_rewards}")

plt.hist(selected_ad)
plt.show()


"""
zeros = [0]*len_col
ones = [0]*len_col
total_rewards = 0

selected_ad = []
for i in range(0,len_row):
    ad = 0
    max_th = 0
    for j in range(0,len_col):
        randbeta = random.betavariate(ones[j]+1,zeros[j]+1)
        if randbeta>max_th:
            max_th = randbeta
            ad = j
    selected_ad.append(ad)
    if df.values[len_row-i-1,ad] == 1: # 10000 -> 0
        ones[ad] +=1
        total_rewards += 1
    else:
        zeros[ad] +=1

print(f"Total Reward : {total_rewards}")

plt.hist(selected_ad)
plt.show()

"""