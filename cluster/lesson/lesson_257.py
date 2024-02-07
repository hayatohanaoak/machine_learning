import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.cluster import KMeans

df = pd.read_csv('./3.01. Country clusters.csv')

# クラスタリング
x      = df.iloc[:, 1:3] # Country, Latitude, Longitude, Language のうち、Latitude, Longitudeを抽出
kmeans = KMeans(3)       # 3つのクラスターに分けるインスタンス
kmeans.fit(x)            # データをクラスタリング
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)
# データフレームを作成
df_with_cluster            = df.copy()
df_with_cluster['Cluster'] = identified_clusters
print(df_with_cluster)

# 散布図を作成
plt.scatter(df['Latitude'], df['Longitude'], c=df_with_cluster['Cluster'], cmap='rainbow')
plt.xlabel('Latitude', fontsize=20)
plt.ylabel('Longitude', fontsize=20)
plt.show()