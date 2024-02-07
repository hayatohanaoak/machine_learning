import pandas as pd
import statsmodels.api as sm

# データの読み込み
df      = pd.read_csv('./real_estate_price_size_year_view.csv')
copy_df = df.copy()
copy_df['view'] = df['view'].map({'No sea view': 0, 'Sea view': 1})
print(copy_df.describe())  # 集計出力

# 従属変数と独立変数の宣言
y  = copy_df['price']
x1 = copy_df[['size', 'year', 'view']]

# 最小二乗回帰
x      = sm.add_constant(x1)
model  = sm.OLS(y, x)
result = model.fit()
report = result.summary()
print(report)  # 回帰結果出力

# データの用意
print('元データ\n',x) # データの確認
new_data = pd.DataFrame({'const': 1, 'size':[300, 600], 'year':2010, 'view':[1, 0]})
pre_data = new_data[['const', 'size', 'year', 'view']]
print('予測したいデータ\n', pre_data)

# 予測
predictions = result.predict(pre_data)
print('予測値\n', predictions)