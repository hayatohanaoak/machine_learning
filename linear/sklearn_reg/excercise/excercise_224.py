from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('./real_estate_price_size_year.csv')
x  = df[['size', 'year']]
y  = df['price']
# 標準化モデルの作成と最適化
scaler_model = StandardScaler()
scaler_model.fit(x)
# 入力値の標準化
std_data = scaler_model.transform(x)
# xを標準化した入力値の重回帰分析
reg_model = LinearRegression()
reg_model.fit(std_data, y)
# 決定係数、切片、係数を算出
r         = reg_model.score(std_data, y)
intercept = reg_model.intercept_
coef_vals = reg_model.coef_
# 自由度修正済み決定係数を算出
n         = x.shape[0]
p         = x.shape[1]
adj_r     = 1 - ( 1 - r ) * ( n - 1 ) / (n - p - 1)
print(r, adj_r)  # 0.7764803683276793 0.77187171612825
"""
決定係数 = 0.7764803683276793
自由度修正済み決定係数 = 0.77187171612825
決定係数 > 自由度修正済み決定係数 となっているが、差は0.1以下である
i.e. 入力値を増やした事による影響は少ない
"""
# 予測データを標準化（標準化モデル作成 -> 最適化 -> 標準化）
new_data          = pd.DataFrame([[750, 2010]], columns=['size', 'year'])  # 予測したいデータ
pred_scaler_model = StandardScaler()
pred_scaler_model.fit(new_data)
std_new_data      = pred_scaler_model.transform(new_data)
# 入力値を学習させたモデル（reg_model）で予測
pred_data = reg_model.predict(std_new_data)
print(pred_data)
# 各入力値のp値の導出
f_p_values = f_regression(x, y)
p_values   = f_p_values[1].round(3)
print(p_values)
# まとめ表の作成
summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
summary['coef-vals'] = coef_vals.reshape(-1, 1)
summary['p-values'] = p_values.round(3)
print(summary)