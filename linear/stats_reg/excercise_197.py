# 不動産に関するデータセットがあります
# 不動産には値段と床面積などに因果関係があることが一般的です
# データは以下のファイルとして保存してあります| 'real_estate_price_size_year.csv'.
# ここで、重回帰分析のモデルを作成してみましょう
# この問題では、従属変数がpriceで独立変数がsizeとyearなります

import pandas as pd
import statsmodels. api as sm

# データの読み込み
df    = pd.read_csv('./real_estate_price_size_year.csv')
total = df.describe()
print(total)
y     = df['price']
x1    = df[['size', 'year']]

# 回帰分析モデル作成
x      = sm.add_constant(x1)
model  = sm.OLS(y, x)
result = model.fit()

# 結果の出力
print(result.summary())