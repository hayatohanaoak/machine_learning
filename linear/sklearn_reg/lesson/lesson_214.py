import pandas as pd
from sklearn.linear_model import LinearRegression

# データの用意
df = pd.read_csv('./1.02.Multiple-linear-regression.csv')

x = df[['Rand 1,2,3', 'SAT']] # 入力値（独立変数）：x
y = df['GPA']                 # 出力値（従属変数）：y

# 重回帰分析の実施
# LinearRegressionは重回帰を前提としているため、今回はデータのreshapeはなし
reg = LinearRegression()
reg.fit(x, y)

# 分析値
coef      = reg.coef_          # 各データの係数
intercept = reg.intercept_     # 切片
print('\n'.join([
    f'coef = { coef }',        # coef = [-0.00826982  0.00165354] intercept=0.2960326126490922
    f'intercept={ intercept }' # coef = [-0.00826982  0.00165354] intercept=0.2960326126490922
]))

#  自由度修正済み決定係数の算出
r  = reg.score(x,y)                            # 決定係数
n  = x.shape[0]                                # データの総数（行）
p  = x.shape[1]                                # データの列数（列）
r2 = 1 - ( 1 - r ) * ( n - 1 ) / ( n - p - 1 ) # 自由度修正済み決定係数
print('\n'.join([
    f'r={ r }',                                # r=0.4066811952814283
    f'r2={ r2 }',                              # r2=0.3920313482513401
    f'r < r2となるか？：{ r < r2}'              # r < r2となるか？：False
]))