import pandas as pd
import numpy as np
import statsmodels.api as sm

# データ下準備
df        = pd.read_csv('./Bank-data.csv')
c_df      = df.copy()
c_df['y'] = c_df['y'].map({'yes': 1, 'no': 0})
# ロジスティック回帰分析
y     = c_df['y']
x1    = c_df['duration']
x     = sm.add_constant(x1)
model = sm.Logit(y, x)
result = model.fit()
# print(result.summary())  # まとめ表の確認
# print(np.exp(0.0051))    # まとめ表で、duration の coef = 0.0051

# 予測の正確性を図る
pred_vals      = result.pred_table()  # 実値 × 予測値の2次元配列を取得
accuracy_train = (pred_vals[0, 0] + pred_vals[1, 1]) / pred_vals.sum()
print(accuracy_train)

# 一応見やすい表も出力
cm_df         = pd.DataFrame(pred_vals)
cm_df.columns = ['予測値 0', '予測値 1']
cm_df         = cm_df.rename(index={0: '実値 0', 1: '実値 1'})
print(cm_df)