from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込みと変換（2次元配列に変換）
df       = pd.read_csv('./real_estate_price_size.csv')
x        = df['size']
y        = df['price']
x_matrix = x.values.reshape(x.shape[0], 1)  # 2次元配列に変換

# モデルの作成
reg   = LinearRegression()
reg.fit(x_matrix, y)

# 各係数などの算出
r         = reg.score(x_matrix, y) # 決定係数
coef      = reg.coef_              # 各変数の係数
intercept = reg.intercept_         # 切片

# グラフの描画
plt.scatter(x, y)
y_hat = coef * x_matrix + intercept
plt.plot(x, y_hat, lw=4, c='blue', label='Regression line')
plt.xlabel('size', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()