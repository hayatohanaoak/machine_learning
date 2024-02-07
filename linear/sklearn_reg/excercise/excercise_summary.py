from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt



# データの確認
df = pd.read_csv('./1.04. Real-life example.csv')
# データクレンジング（列・欠損値のあるレコードの削除）
df_drop_col = df.drop(['Model', 'Registration'], axis=1) # 列を削除（axis=0で行・axis=1で列）
df_no_mv    = df_drop_col.dropna(axis=0)                 # 欠損値のあるレコードを削除
# データクレンジング（外れ値の削除）
q_1  = df_no_mv['Price'].quantile(0.99)  # データの最小値から99%の値を取得
df_1 = df_no_mv[df_no_mv['Price'] < q_1] # 取得値を閾値にすることで、外れ値を除外
q_2  = df_1['Mileage'].quantile(0.99)
df_2 = df_1[df_1['Mileage'] < q_2]
q_3  = df_2['EngineV'].quantile(0.99)
df_3 = df_2[df_2['EngineV'] < q_3]
q_4  = df_3['Year'].quantile(0.01)  # Yearだけは下に外れ値なので、
df_4 = df_3[df_3['Year'] > q_4]     # 下位1%を閾値にする
# データの再作成
cleaned_df = df_4.reset_index(drop=True) # インデックスの振り直し（drop=Trueで古いインデックスは削除）



# 出力値と各入力値の間における線形性を、散布図で観察
_, before_trance_axs = plt.subplots( # 描画フィールドの作成
    nrows=1,        # フィールド行
    ncols=3,        # フィールド列
    sharey=True,    # 縦軸と横軸の値を固定
    figsize=(15, 3) # 横幅と縦幅のインチ指定
)
before_trance_axs[0].scatter(cleaned_df['Year'], cleaned_df['Price'])
before_trance_axs[0].set_title('Log Price and Year')
before_trance_axs[1].scatter(cleaned_df['EngineV'], cleaned_df['Price'])
before_trance_axs[1].set_title('Log Price and EngineV')
before_trance_axs[2].scatter(cleaned_df['Mileage'], cleaned_df['Price'])
before_trance_axs[2].set_title('Log Price and Mileage')
plt.show()



# 対数変換したデータを追加
log_price = np.log(cleaned_df['Price'])
cleaned_df['log_price'] = log_price



# 変換後の出力値と各入力値の間における線形性を、散布図で観察
_, after_trance_axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 3))
after_trance_axs[0].scatter(cleaned_df['Year'], cleaned_df['log_price'])
after_trance_axs[0].set_title('Log Price and Year')
after_trance_axs[1].scatter(cleaned_df['EngineV'], cleaned_df['log_price'])
after_trance_axs[1].set_title('Log Price and EngineV')
after_trance_axs[2].scatter(cleaned_df['Mileage'], cleaned_df['log_price'])
after_trance_axs[2].set_title('Log Price and Mileage')
plt.show()



# 多重共線性の評価
variables  = cleaned_df[['Mileage', 'Year', 'EngineV']]
vif        = pd.DataFrame()
vif['VIF'] = [
    variance_inflation_factor(variables.values, i)
    for i in range(variables.shape[1]) # データフレームの行数で繰り返し
]
vif['features'] = variables.columns
print(vif)
# 不要と判断した列を削除
df_no_multi_col_linearity = cleaned_df.drop(['Year', 'Price'], axis=1)



# カテゴリ変数をダミー変数に変換
df_with_dummys = pd.get_dummies(
    df_no_multi_col_linearity, # 変換元データ
    drop_first=True,           # カテゴリ変数の数-1 のダミー変数を作成するオプション値
    dtype=int                  # 変換後のデータ型を指定（デフォオルトだとTrue、Falseで変換される！）
)
print(df_with_dummys.columns.values)
print(df_with_dummys)
COLS = [
    'log_price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
    'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol'
]
df_preprocessed = df_with_dummys[COLS]  # 指定順に並べ替え



# 課題1：data_preprocesedの全ての変数が含まれている状態で、多重共線性の評価
hw_vif_1             = pd.DataFrame()
hw_vif_1['VIF']      = [
    variance_inflation_factor(df_preprocessed.values, i)
    for i in range(df_preprocessed.shape[1])
]
hw_vif_1['features'] = df_preprocessed.columns
print(hw_vif_1)
# 課題2：従属変数（log_price）を除いて計算
df_variable_1   = df_with_dummys.drop(['log_price'], axis=1)
hw_vif_2        = pd.DataFrame()
hw_vif_2['VIF'] = [
    variance_inflation_factor(df_variable_1.values, i)
    for i in range(df_variable_1.shape[1])
]
hw_vif_2['features'] = df_variable_1.columns
print(hw_vif_2)



# 入力値と出力値に分けて、入力値は標準化
inputs  = df_preprocessed.drop(['log_price'], axis=1)
targets = df_preprocessed['log_price']
scaler  = StandardScaler()
scaler.fit(inputs)
input_scaled = scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(input_scaled, targets, test_size=0.2, random_state=365)



# 回帰モデル作成
model = LinearRegression()
model.fit(x_train, y_train)
y_hat = model.predict(x_train)  # 値の予測



# グラフ描画（入力値と予測値）
plt.scatter(y_train, y_hat)
plt.xlabel('Targets(y_train)', size=18)
plt.ylabel('predictions(y_hat)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()
# グラフ描画（入力値と予測値の残差）
sns.displot(y_train - y_hat)
plt.title('Residuals PDF', size=18)
plt.show()



# 決定係数・切片・定数の算出
model_score = model.score(x_train, y_train)
intercept   = model.intercept_
coef        = model.coef_
# 見やすく成形して出力
summary         = pd.DataFrame(inputs.columns.values, columns=['Features'])
summary['coef'] = coef
print(f'model_score={model_score}, intercept={intercept}')
print(summary)



# 実際のデータの予測
y_hat_test = model.predict(x_test)
# グラフ描画（入力値と予測値の関係・透過あり）
plt.scatter(y_test, y_hat_test, alpha=0.3)
plt.xlabel('Targets(y_train)', size=18)
plt.ylabel('predictions(y_hat)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()



# 対数価格の指数をとって、元の価格に変換（価格 = x e^x => log e^p = p）
df_pf           = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions']) # 予測値
df_pf['Target'] = np.exp(y_test)    # 実データ
print(df_pf.head()) # NaNが混じる。原因は下にある通り。
print(y_test)       # テストデータ作成時の抽出データのため、ランダムなインデックスの列がある事を確認…
# インデックスの列を消し、新たにインデックスが連番で振られる
r_y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(r_y_test)
print(df_pf.head())




# 予測値と実値の残差（Residual）を求め、残差の割合（Difference%）のデータ求め、それぞれ追加する
df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']                    
df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100) # 残差を%で表す
print(df_pf)
print(df_pf.describe())
# pandasの設定を変更（999列表示・小数第2位まで表示）
pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df_pf.sort_values(by=['Difference%'])) # Difference%でソートして表示