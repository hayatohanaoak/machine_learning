from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# データの確認
df = pd.read_csv('./iris_dataset.csv')
_, confirm_axes = plt.subplots(  # 描画フィールドの作成
    nrows=1,
    ncols=2,
    sharey=True,
    figsize=(15, 3)
)
confirm_axes[0].scatter(df['sepal_length'], df['sepal_width'])
confirm_axes[0].set_title('x = sepal_length, y = sepal_width')
confirm_axes[1].scatter(df['petal_length'], df['petal_width'])
confirm_axes[1].set_title('x = petal_length, y = petal_width')
plt.show()

# データを標準化して格納
df_scaled = preprocessing.scale(df)

# クラスターが2, 3, 5の場合を検証（まとめて予測・まとめて描画・まとめて出力）
_, result_axes = plt.subplots(
    nrows=2,
    ncols=3,
    sharey=True,
    figsize=(15, 3)
)
for idx, j in enumerate([2, 3, 5]):
    
    kmeans = KMeans(j)
    kmeans.fit(df_scaled)
    cluster = kmeans.fit_predict(df_scaled)
    
    result_axes[0][idx].scatter(df['sepal_length'], df['sepal_width'], c=cluster, cmap='rainbow')
    result_axes[0][idx].set_title(f'{j} clusters plot(x = sepal_length, y = sepal_width)')
    result_axes[1][idx].scatter(df['petal_length'], df['petal_width'], c=cluster, cmap='rainbow')
    result_axes[1][idx].set_title(f'{j} clusters plot(x = petal_length, y = petal_width)')
    
    c_df = df.copy()
    c_df['cluster'] = cluster
    c_df.to_csv(f'./iris_with_answer_cluster{j}.csv', index=False) # インデックスなしで出力
plt.show()

# エルボー法の確認
wcss = []
for i in range(1, 6):
    kmeans_iter = KMeans(i)
    kmeans_iter.fit(df_scaled)
    wcss.append(kmeans_iter.inertia_)
plt.plot(range(1,6), wcss)
plt.show()