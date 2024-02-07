import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# データの用意
df = pd.read_csv('./1.02.Multiple-linear-regression.csv')

x = df[['Rand 1,2,3', 'SAT']]
y = df['GPA']

# 重回帰分析の実施
reg = LinearRegression()
reg.fit(x, y)

# 分析値
coef      = reg.coef_          # 各データの係数
intercept = reg.intercept_     # 切片

#  自由度修正済み決定係数の算出
r  = reg.score(x,y)                            # 決定係数
n  = x.shape[0]                                # データの総数（行）
p  = x.shape[1]                                # データの列数（列）
r2 = 1 - ( 1 - r ) * ( n - 1 ) / ( n - p - 1 ) # 自由度修正済み決定係数

# 単回帰による各独立変数のF値とP値の検証
f_p_values = f_regression(x, y)
p_values = f_p_values[1]

# まとめ表の作成
reg_summary = pd.DataFrame(
    data = x.columns.values, columns=['features']) # 列名を羅列した行を生成する
reg_summary['p-values'] = p_values.round(3)
print(reg_summary)
