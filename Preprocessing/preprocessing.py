from pandas import DataFrame
import statsmodels.api as sm

class LabelEncoder:

    def fit_transform(self, df:DataFrame(), columns:list):
        temp_df = df.copy()
        for col in columns:
            uniq = temp_df[col].unique()
            for i in range(len(uniq)):
                temp_df[col] = temp_df[col].replace(uniq[i],i)
        return temp_df


class OneHotEncoder:

    def fit_transform(self,df:DataFrame(),columns:list):
        temp_df = df.copy()
        for col in columns:
            uniq = temp_df[col].unique()
            if len(uniq)==2:
                for i in range(2):
                    temp_df[col] = temp_df[col].replace(uniq[i],i)
            else:
                for item in uniq:
                    new_column = [int(i) for i in df[col] == item]
                    temp_df[item] = new_column
                del temp_df[col]

        return temp_df


def backward_elimination(data: DataFrame(), y, significance=0.05):
    temp_df = data.copy()
    index_list = [*range(0, len(temp_df.columns))]

    while True:
        X = temp_df.iloc[:, index_list].values
        model = sm.OLS(y, X).fit()
        finish = True
        for i in range(len(index_list)):
            if model.pvalues[i] >= significance:
                index_list.__delitem__(i)
                finish = False
        if finish:
            break
    for i in range(max(index_list)):
        if not index_list.__contains__(i):
            temp_df.drop(temp_df.columns[i], axis=1, inplace=True)
    return temp_df
