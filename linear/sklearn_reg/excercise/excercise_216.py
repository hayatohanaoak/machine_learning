from sklearn.linear_model import LinearRegression
from linear_reg_total import adj_2r
import pandas as pd

df = pd.read_csv('./1.02.Multiple-linear-regression.csv')
x  = df[['SAT', 'Rand 1,2,3']]
y  = df['GPA']

model = LinearRegression()
model.fit(x, y)

coef      = model.coef_
intercept = model.intercept_
totals    = adj_2r(x, y)
r2        = totals['adj_r_score']
print(r2)