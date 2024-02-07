import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('./2.02. Binary predictors.csv')

# ロジスティック回帰の準備（データ整形）
c_df             = df.copy()
c_df['Admitted'] = df['Admitted'].map({'Yes': 1, 'No': 0})
c_df['Gender']   = df['Gender'].map({'Female': 1, 'Male': 0})
# ロジスティック回帰
y      = c_df['Admitted']
x1     = c_df[['SAT', 'Gender']]
x      = sm.add_constant(x1)
model  = sm.Logit(y, x)
result = model.fit()
# print(result.summary())

# 予測の正確性を図る
pred_vals = result.pred_table()  # 実値 × 予測値の2次元配列を取得
print(pred_vals)
cm_df         = pd.DataFrame(pred_vals)
cm_df.columns = ['予測値 0', '予測値 1']
cm_df         = cm_df.rename(index={0: '実値 0', 1: '実値 1'})
print(cm_df)

# 予測の的中率計算
accuracy_train = (pred_vals[0][0] + pred_vals[1][1]) / pred_vals.sum()
print(accuracy_train)