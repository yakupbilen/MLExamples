from apriori import apriori
shop = []

with open("shop.csv") as file:
    shop = file.readlines()

shop = [line.replace("\n","") for line in shop]
data = []
for line in shop:
    temp = []
    for item in line.split(","):
        temp.append(item)
    data.append(temp)

rules = apriori(data,min_support=0.01,min_confidence=0.03,min_lift=2,min_length=2)
print(list(rules))

