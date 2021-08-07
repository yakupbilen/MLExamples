from pandas import DataFrame


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
