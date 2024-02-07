from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('./1.02.Multiple-linear-regression.csv')
x  = df[['SAT', 'Rand 1,2,3',]]
y  = df['GPA']

# 標準化モジュールを使って、xを標準化
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x[:10])
print(x_scaled[:10])

# 標準値を用いたモデルの作成と、係数・切片の表作成
model = LinearRegression()
model.fit(x_scaled, y)
reg_summary = pd.DataFrame([['Bias'],['SAT'],['Rand 1,2,3']], columns=['Features'])
reg_summary['Weights'] = model.intercept_, model.coef_[0], model.coef_[1]
print(reg_summary)

# 予測
new_data        = pd.DataFrame([[1700, 2],[1800, 1]], columns=['SAT', 'Rand 1,2,3'])
new_data_scaled = scaler.transform(new_data)     # 新規データも標準化（モデル作成時と同じ型にする！）
pred_data       = model.predict(new_data_scaled)
print(f'Rand 1,2,3あり：{pred_data}')

# Rand 1,2,3を取り除く 手順：単回帰分析モデルの作成 -> 予測 -> 値の出力と比較
simple_model   = LinearRegression()
x_simple_model = x_scaled[:, 0].reshape(-1, 1)  # SATのデータを二次元配列に変換
simple_model.fit(x_simple_model, y)
simple_pred    = simple_model.predict(new_data_scaled[:, 0].reshape(-1, 1))
print(f'Rand 1,2,3なし：{simple_pred}')