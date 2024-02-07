import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    カスタム標準化クラス
    """
    def __init__(self, columns) -> None:
        self.scaler  = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.columns = columns
        self.mean_   = None
        self.var_    = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_  = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


class absenteeismModel():
    """
    モデルを使用するクラス
    """
    def __init__(self, model_file, scale_file):
        with open(model_file, 'rb') as mf, open(scale_file, 'rb') as sf:
            self.reg = pickle.load(mf)
            self.scaler = pickle.load(sf)
            self.data = None

    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter=',')
        self.df_with_predictions = df.copy()  # 元データの保持

        df = df.drop(['ID'], axis=1)
        df['Absenteeism Time in Hours'] = ''
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        # 欠席理由の変換
        reason_dummies = pd.get_dummies(df['Reason for Absence'], drop_first=True) # n-1個のdummy変数を作成
        print(reason_dummies.iloc[:,15:17])
        reason_group1  = reason_dummies.loc[:,  1:14].max(axis=1).map({False: 0, True: 1})
        reason_group2  = reason_dummies.loc[:, 15:17].max(axis=1).map({False: 0, True: 1})
        reason_group3  = reason_dummies.loc[:, 18:21].max(axis=1).map({False: 0, True: 1})
        reason_group4  = reason_dummies.loc[:, 22:  ].max(axis=1).map({False: 0, True: 1})
        df             = df.drop(['Reason for Absence'], axis=1)
        df             = pd.concat([df, reason_group1, reason_group2, reason_group3, reason_group4], axis=1)
        df.columns     = list(df.columns.values[:-4]) + ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']  # 結合した後ろ4つのカラム名を変更

        # 日付データの整形
        df['Date']             = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Month Value']      = df['Date'].apply(lambda date_val: date_val.month)
        df['Day of the Week']  = df['Date'].apply(lambda date_val: date_val.weekday())
        df                     = df.drop(['Date'], axis=1)
        columns_names_recorded = [
            'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value' ,'Day of the Week', 'Transportation Expense' ,'Distance to Work',
            'Age', 'Daily Work Load Average', 'Body Mass Index' ,'Education', 'Children', 'Pets', 'Absenteeism Time in Hours'
        ]
        # 整列
        df = df[columns_names_recorded]
        """
        df_with_targetsで不要な理由
        Absenteeism Time in Hours
            今回は、欠席するか否かなので、欠席時間（欠席の度合い）は不問
        Distance to Work
            一度分析をした結果、このデータが与える影響が少ない（オッズが0.98と1に限りなく近い）ため
        Daily Work Load Average
            一度分析をした結果、このデータが与える影響が少ない（オッズが0.98と1に限りなく近い）ため
            データの中身を見ると、データ間にそこまでの大きな差がないため
        """
        df = df.drop([
            'Absenteeism Time in Hours', 'Distance to Work', 'Daily Work Load Average'], axis=1)
        df = df.fillna(value=0)
        self.preprocessed_data = df.copy()
        self.data = self.scaler.transform(df)
        
        return df

    @property
    def predicted_probability(self):
        if self.data is not None:
            pred = self.reg.predict_proba(self.data)[:, 1]

            return pred

    @property
    def predicted_output_category(self):
        if self.data is not None:
            pred_output = self.reg.predict(self.data)

            return pred_output

    @property
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = \
                self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = \
                self.reg.predict(self.data)

            return self.preprocessed_data