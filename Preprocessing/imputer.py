from pandas import DataFrame
from numpy import nan, isnan
from nltk import FreqDist


class Imputer():

    def __init__(self, missing_values=nan, strategy="mean"):
        self.missing_values = missing_values
        self.strategy = strategy
        self.impute_values = []
        self.columns = []

    def fit(self, df: DataFrame(), columns: list):
        self.impute_values.clear()
        self.columns.clear()
        self.df = df
        if self.strategy == "mean":
            for column in columns:
                if isinstance(self.missing_values, str):
                    temp = df[df[column] != self.missing_values].copy()
                else:
                    temp = df[df[column].notna()].copy()
                try:
                    temp[column] = temp[column].astype(str).astype(float)
                    total = temp[column].sum()
                    mean = total / temp.shape[0]
                    self.impute_values.append(mean)
                    self.columns.append(column)

                except ValueError:
                    raise ValueError("Column to fill is NOT int or float!!!")

        elif self.strategy == "median":
            for column in columns:
                if isinstance(self.missing_values, str):
                    temp_df = df[df[column]!=self.missing_values].copy()
                else:
                    temp_df = df[df[column].notna()].copy()

                try:
                    temp_df[column] = temp_df[column].astype(str).astype(float)
                except ValueError:
                    raise ValueError("Column to fill is NOT int or float!!!")
                if temp_df[column].dtypes == "float64":
                    item_count = temp_df.shape[0]
                    if item_count % 2 == 0:
                        temp_df = temp_df.sort_values(by=column)
                        median = temp_df.iloc[int(item_count / 2)][column]
                        median = median + temp_df.iloc[int(item_count / 2) - 1][column]
                        median = median / 2

                    else:
                        temp_df = df.sort_values(by=column)
                        median = temp_df.iloc[int(item_count / 2)][column]

                    self.impute_values.append(median)
        elif self.strategy == "most_frequent":
            for column in columns:
                temp_df = df[df[column].notna()]
                fdist = FreqDist(samples=temp_df[column])
                common = fdist.most_common(1)
                self.impute_values.append(common[0][0])
                self.columns.append(column)

        else:
            raise ValueError("Strategy must be mean,median or most_frequent")

    def transform(self):
        i = 0
        temp_df = self.df.copy()
        for column in self.columns:
            try:
                temp_df[column] = temp_df[column].replace(self.missing_values,self.impute_values[i]).astype(str).astype(float)
            except ValueError:
                temp_df[column] = temp_df[column].replace(self.missing_values, self.impute_values[i])
            i += 1
        return temp_df
