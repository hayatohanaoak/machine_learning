import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('./Example-bank-data.csv')

# ロジスティック回帰
c_df      = df.copy()
c_df['y'] = c_df['y'].map({'yes': 1, 'no': 0})  # カテゴリ変数のデータを、パラメータに変換
y         = c_df['y']         # 出力値
x1        = c_df['duration']  # 入力値
x         = sm.add_constant(x1)
model     = sm.Logit(y, x)    # ロジスティック回帰モデル作成
result    = model.fit()       # 最適化
print(result)
print(result.summary())

# 散布図の作成
plt.scatter(x1, y)
plt.xlabel('duration', fontsize=20)
plt.ylabel('subscript', fontsize=20)
plt.show()