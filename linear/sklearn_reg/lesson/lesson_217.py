import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression  # それぞれの変数における単線形回帰を行う

# データの用意
df = pd.read_csv('./1.02.Multiple-linear-regression.csv')
x = df[['Rand 1,2,3', 'SAT']]
y = df['GPA']
# 単回帰による各独立変数のF値とP値の検証
f_p_values = f_regression(x, y)
print(f"f_p_values = {f_p_values}")
p_values = f_p_values[1]
print(f"['Rand 1,2,3', 'SAT'] = {p_values}")
print(f"['Rand 1,2,3', 'SAT'] = {p_values.round(3)}")