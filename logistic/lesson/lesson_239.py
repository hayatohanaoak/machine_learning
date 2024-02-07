import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import statsmodels.api as sm


df                  = pd.read_csv('./2.01. Admittance.csv')
copy_df             = df.copy()
copy_df['Admitted'] = copy_df['Admitted'].map({'Yes': 1, 'No': 0})

# ロジスティック回帰分析
y      = copy_df['Admitted']
x1     = copy_df['SAT']
x      = sm.add_constant(x1)
model  = sm.Logit(y, x)
result = model.fit()
print(result)
# y_hat  = result.params[0] + result.params[1]*x1

# おかしなグラフになっている事を確認（直線も点もおかしい）
# plt.scatter(x1, y, color='C0')
# plt.plot(x1, y_hat, lw = 2.5, color='C8')
# plt.xlabel('SAT', fontsize=20)
# plt.ylabel('Admitted', fontsize=20)
# plt.show()