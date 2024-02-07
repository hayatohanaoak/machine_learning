import pandas as pd
import numpy as np
import statsmodels.api as sm

# 混同行列の作成メソッド
def confusion_matrix(data, actual_vals, model):
    pred_vals = model.predict(data)
    bins      = np.array([0, 0.5, 1])
    cm        = np.histogram2d(actual_vals, pred_vals, bins=bins)[0]
    accuracy  = ( cm[0, 0] + cm[1, 1] ) / cm.sum()
    return accuracy

df = pd.read_csv('./2.02. Binary predictors.csv')

# ロジスティック回帰の準備（データ整形）
c_df             = df.copy()
c_df['Admitted'] = df['Admitted'].map({'Yes': 1, 'No': 0})
c_df['Gender']   = df['Gender'].map({'Female': 1, 'Male': 0})
# ロジスティック回帰
y      = c_df['Admitted']
x1     = c_df[['SAT', 'Gender']]
x      = sm.add_constant(x1)
model  = sm.Logit(y, x)
result = model.fit()

# 9:1で分けたデータを変換して、使ってテスト
test             = pd.read_csv('./2.03. Test dataset.csv')
test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No': 0})
test['Gender']   = test['Gender'].map({'Female': 1, 'Male': 0})
test_actual      = test['Admitted']
test_x           = test[['SAT', 'Gender']]
test_data        = sm.add_constant(test_x)

if x.columns.values.all() == test_data.columns.values.all():  # 2つのデータが同じ並びをしていることを確認
    cm = confusion_matrix(test_data, test_actual, result)
    print(cm)
else:
    print('2データの列の並びを揃えてください')