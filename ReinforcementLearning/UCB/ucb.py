import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv("../Ads_CTR_Optimisation.csv")
print(df.head(5))

len_col = len(df.columns)
len_row = len(df)

number_of_clicking = [0]*len_col
sum_of_rewards = [0]*len_col
total_rewards = 0

selected_ad = []

for i in range(0,len_row):
    ad = 0
    max_ucb = 0
    for j in range(0,len_col):
        if number_of_clicking[j]>0:
            average = sum_of_rewards[j]/number_of_clicking[j]
            delta = math.sqrt(3/2 * math.log(i+1)/number_of_clicking[j])
            bound = delta+average
        else:
            bound = 2
        if bound>max_ucb:
            max_ucb = bound
            ad = j
    selected_ad.append(ad)
    number_of_clicking[ad] +=1
    sum_of_rewards[ad] += df.values[i,ad] # 1.Day -> 10,000.Day
    total_rewards += df.values[i,ad]

print(f"Total Reward : {total_rewards}")


plt.hist(selected_ad)
plt.show()

"""
number_of_clicking = [0]*len_col
sum_of_rewards = [0]*len_col
total_rewards = 0

selected_ad = []

for i in range(0,len_row):
    ad = 0
    max_ucb = 0
    for j in range(0,len_col):
        if number_of_clicking[j]>0:
            average = sum_of_rewards[j]/number_of_clicking[j]
            delta = math.sqrt(3/2 * math.log(len_row-i)/number_of_clicking[j]) # 10,000.Day -> 1.Day
            bound = delta+average
        else:
            bound = 10
        if bound>max_ucb:
            max_ucb = bound
            ad = j
    selected_ad.append(ad)
    number_of_clicking[ad] +=1
    sum_of_rewards[ad] += df.values[len_row-i-1,ad] # 10,000.Day -> 1.Day
    total_rewards += df.values[len_row-i-1,ad] # 10,000.Day -> 1.Day

print(f"Total Reward : {total_rewards}")


plt.hist(selected_ad)
plt.show()
"""