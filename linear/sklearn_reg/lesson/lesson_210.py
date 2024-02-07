import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# データの用意
df = pd.read_csv('./1.01.Simple-linear-regression.csv')
x1 = df['SAT']
y  = df['GPA']
# データの適応
x_matrix = x1.values.reshape(84, 1)

# 回帰モデルの作成
reg   = LinearRegression()
model = reg.fit(x_matrix, y)

# 分析値の算出
r_squared    = reg.score(x_matrix, y)  # 決定係数の算出
coefficients = reg.coef_               # 各係数の算出（list）
intercept    = reg.intercept_          # 切片
print(
    f'r_squared = {r_squared}',
    f'coefficients={coefficients}',
    f'intercept={intercept}'
)

# 予測
pre_value = reg.predict([[1740]])
print(f'pre_value={pre_value}')

# 予測（pandasのデータフレームを渡すVer）
new_data     = pd.DataFrame(data=[1740, 1760], columns=['SAT'])  # 第一引数がリスト、第二引数がデータフレームなので、warningが出る…
df_pre_value = reg.predict(new_data)
new_data['predict_GPA'] = df_pre_value
print(f'new_data=\n{new_data}')

# グラフの描画
plt.scatter(x1, y)
y_hat = reg.coef_ * x1.values.reshape(84, 1) + reg.intercept_
fig = plt.plot(x1, y_hat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()