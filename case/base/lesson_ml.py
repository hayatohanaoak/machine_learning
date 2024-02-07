import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# データ取得と出力値の変換
df_preprocessed = pd.read_csv('./df_remake_complete.csv')
ath             = df_preprocessed['Absenteeism Time in Hours']
targets         = np.where(
    ath > ath.median(), # 条件式
    1,                  # 真の時
    0                   # 偽の時
)
df_preprocessed['Excessive Absenteeism'] = targets
average = targets.sum() / targets.shape[0]
# print(average)  # およそ0.46なので、0と1が大体半々の割合になっている

# 入力値の設定
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
df_with_targets = df_preprocessed.drop(
    ['Absenteeism Time in Hours', 'Distance to Work', 'Daily Work Load Average'], axis=1)
unscaled_inputs = df_with_targets.iloc[:, :-1]  # Absenteeism Time in Hours以外

# データの標準化（ダミー変数以外）
absenteeism_scaler = StandardScaler()
absenteeism_scaler.fit(unscaled_inputs)
class CustomScaler(BaseEstimator, TransformerMixin):
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
col_to_omit = { # 不要カラムの集合
    'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Distance to Work', 'Daily Work Load Average'
}
unscaled_to_scale = list(
    set(unscaled_inputs.columns.values) - col_to_omit  # 既存カラムと不要カラムの差集合
)

absenteeism_scaler = CustomScaler(unscaled_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# 訓練用・テスト用に分割し、実行毎のデータシャッフルを無効に設定
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)

# ロジスティック回帰分析
reg = LogisticRegression()
reg.fit(x_train, y_train)
train_score = reg.score(x_train, y_train)
# print(f'train_score = {train_score}')
model_outputs = reg.predict(x_train)

# 係数データの取得と作成
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['feature_name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table.index += 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()

# オッズ（倍率）データの取得
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
check_table = summary_table.sort_values('Odds_ratio', ascending=False) # 降順に並び替え

# モデルのテスト
predicted_score = reg.score(x_test, y_test)
predicted_proba = reg.predict_proba(x_test)
# print(f'pred_score = {predicted_score}')
# print(f'predict_proba = {predicted_proba}')  # [0になる確率, 1になる確率]

# モデルの保存
with open('../model', 'wb') as f:
    pickle.dump(reg, f)  # ロジスティック回帰モデルの保存
f.close()
with open('../scaler', 'wb') as f:
    pickle.dump(absenteeism_scaler, f) # 標準化モデルの保存
f.close()
