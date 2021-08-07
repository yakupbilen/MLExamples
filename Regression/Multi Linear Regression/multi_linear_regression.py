import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
from importlib import reload
from Preprocessing.outlier import delete_outliers_qr,delete_outliers_z


def backward_elimination(data: pd.DataFrame(), target):
    temp_df = data.copy()
    index_list = [*range(0, len(temp_df.columns))]

    while True:
        X = temp_df.iloc[:, index_list].values
        model = sm.OLS(target, X).fit()
        finish = True
        for i in range(len(index_list)):
            if model.pvalues[i] >= 0.05:  # Significance level
                index_list.__delitem__(i)
                finish = False
        if finish:
            break
    for i in range(max(index_list)):
        if not index_list.__contains__(i):
            temp_df.drop(temp_df.columns[i], axis=1, inplace=True)
    return temp_df




print("Model 1 = Backward Elimination -> Regression")
print("Model 2 = Deleting outliers with Z-Score -> Backward Elimination -> Regression")
print("Model 3 = Deleting outliers with Interquartile(IQR) -> Backward Elimination -> Regression")


df = pd.read_csv("USA_Housing.csv")

del df["Address"]

df_qr = delete_outliers_qr(df, ["Price"])
df_z = delete_outliers_z(df, ["Price"])
df_list = [df, df_z, df_qr]
i = 1
for dataframe in df_list:
    price = dataframe["Price"]
    max_price = price.max()
    min_price = price.min()
    price = price.apply(lambda x: ((x - min_price) / (max_price - min_price)))  # Min-Max Norm
    del dataframe["Price"]
    backward_elimination(dataframe, price)

    x_train, x_test, y_train, y_test = train_test_split(dataframe, price, test_size=0.3, random_state=0)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    y_test_np = y_test.values

    print(50*"-")
    print(f"Model {i} R2 Score : {r2_score(y_pred, y_test)}")
    print(f"Model {i} Mean Squared Error : {mean_squared_error(y_pred, y_test)}")
    print(f"Model {i} Root Mean Squared Error : {math.sqrt(mean_squared_error(y_pred, y_test))}")
    print(50*"-")
    # Revert minmax normalization back to original values
    result = pd.DataFrame({"Predict Value": (y_pred * (max_price - min_price) + min_price),
                           "Real Value": (y_test_np * (max_price - min_price) + min_price)})

    result.to_csv(f"Model {i} result.csv", index=False)
    i += 1
