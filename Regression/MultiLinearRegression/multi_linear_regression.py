import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
from Preprocessing.outlier import delete_outliers_qr,delete_outliers_z
from Preprocessing.preprocessing import backward_elimination



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

    r2 = r2_score(y_pred, y_test)
    adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - 1 - x_test.shape[1]))
    print(50*"-")
    print(f"Model {i} R2 Score : {r2}")
    print(f"Model {i} Adjusted R2 Score : {adj_r2}")
    print(f"Model {i} Mean Squared Error : {mean_squared_error(y_pred, y_test)}")
    print(f"Model {i} Root Mean Squared Error : {math.sqrt(mean_squared_error(y_pred, y_test))}")
    print(50*"-")
    # Revert minmax normalization back to original values
    result = pd.DataFrame({"Predict Value": (y_pred * (max_price - min_price) + min_price),
                           "Real Value": (y_test_np * (max_price - min_price) + min_price)})

    result.to_csv(f"Model {i} result.csv", index=False)
    i += 1
