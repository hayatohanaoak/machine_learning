from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import pandas as pd

# データ読み込みとモデル作成
df = pd.read_csv('./real_estate_price_size_year.csv')
x  = df[['size', 'year']]
y  = df['price']
model = LinearRegression()
model.fit(x, y)

# 決定係数と自由度修正済み決定係数の算出
r     = model.score(x, y)
n     = x.shape[0]        # 行数
p     = x.shape[1]        # 列数
adj_r = 1 - ( 1 - r ) * ( n - 1 ) / ( n - p - 1 )
print(f'決定係数={ r }, 自由度修正済み決定係数={ adj_r }')
"""
単回帰の場合との比較
決定係数=0.7764803683276793, 自由度修正済み決定係数=0.77187171612825
今回は、決定係数 > 自由度修正済み決定係数 となるが、値に大差はない。（0.1以上の差となると、差が大きい事になる）
-> 今回、独立変数を増やしたことによる分析へのペナルティは少ない。
"""

# 各データの係数と切片の算出
coef      = model.coef_
intercept = model.intercept_
print(f'各係数={ coef }, 切片={ intercept }')

# 予測
price_predict = model.predict([[750, 2013]])
print(f'price_predict = {price_predict}')  # price_predict = [269997.4859673]

# p値の計算（各独立変数）
f_p_values = f_regression(x, y)
p_values   = f_p_values[1].round(3)
print(f'p_values={ p_values }')    # p_values=[0.    0.357]  どちらも0.5以下なので、ペナルティは少ない（無い）

# まとめ表の作成
summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
summary['coef']    = coef
summary['p_value'] = p_values
print(summary)