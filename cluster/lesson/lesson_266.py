from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

df = pd.read_csv('./3.12. Example.csv')
c_df = df.copy()
# データの標準化（Standard Scalarを用いない）
df_scaled = preprocessing.scale(c_df)

# 最適なクラスターの数を探る…（エルボー法）
wcss_vals = []
for i in range(1, 10):
    kmeans_iter = KMeans(i)
    kmeans_iter.fit(c_df)
    wcss_vals.append(kmeans_iter.inertia_)
plt.plot(range(1, 10), wcss_vals)
plt.xlabel('Number of cluster')
plt.ylabel('Wcss')
plt.show()

# クラスタリング
# エルボー法のグラフ的に、クラスターは4つがちょうど良さそう
kmeans = KMeans(4)
kmeans.fit(df_scaled)
cluster = kmeans.fit_predict(df_scaled)

# 描画
c_df['Cluster'] = cluster  # 標準化前のデータフレームに格納
print(c_df)
plt.scatter(c_df['Satisfaction'], c_df['Loyalty'], c=c_df['Cluster'], cmap='rainbow')
plt.xlabel('Satisfaction', fontsize=20)
plt.ylabel('Loyalty', fontsize=20)
plt.show()