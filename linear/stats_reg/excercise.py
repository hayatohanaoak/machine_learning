# 不動産に関するデータセットがあります
# 不動産には値段と床面積などに因果関係があることが一般的です
# データは以下のファイルとして保存してあります 'real_estate_price_size.csv'.
# ここで、単線形回帰を作成してみましょう
# この問題では、従属変数がpriceで独立変数がsizeとなります

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

"""
問1：データの確認
"""
# データの読み込み
df = pd.read_csv(
    './real_estate_price_size.csv',
    encoding='utf-8'
)
y  = df['price']
x1 = df['size']

# データの代表値の確認
total = df.describe()
print(total)

# グラフを描画
plt.scatter(x1, y)
plt.ylabel('price', fontsize=20)
plt.xlabel('size' , fontsize=20)
plt.show()


"""
問2：回帰分析の実施
"""
# 回帰分析の実施
x      = sm.add_constant(x1) # 独立変数を新たにstatsmodelsに追加
model  = sm.OLS(y, x)        # 回帰分析を実施して、モデル作成
result = model.fit()         # 分析値の中でも、最も最適な値の取得
report = result.summary()    # 分析のサマリーレポートを取得
print(report)


"""
問3：回帰分析の結果を受けて、回帰直線を描画
"""
# 回帰分析の結果を、グラフに描画
plt.scatter(x1, y)
# 一般回帰式（y：目的変数、xi：説明変数）： y = Σ(∞, n=0){ bn * xn } ※ x0 = 1
# 今回は、説明変数は1つなので、y = b0 + b1 * x1 = b1 * x1 + b0
y_hat = 223.1787 * x1 + 1.019e+05
plt.plot(
    x1, y_hat, lw=4, c='orange', label='region line'
) # 点の描画
plt.ylabel('price', fontsize=20)
plt.xlabel('size' , fontsize=20)
plt.show()