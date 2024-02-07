from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# データの確認
df                  = pd.read_csv('./Categorical.csv')
target              = df.iloc[:, 1:4]  # Longitude, Latitude, continentを抽出
target['continent'] = target['continent'].map({
    'North America': 0, 'Asia': 1, 'Africa': 2, 'Europe': 3, 'South America': 4,
    'Oceania': 5, 'Antarctica':6, 'Seven seas (open ocean)': 7
})

# クラスタリング
kmeans = KMeans(4)
kmeans.fit(target)
cluster = kmeans.fit_predict(target)
target['Cluster'] = cluster

# 描画
plt.scatter(target['Longitude'], target['Latitude'], c=target['Cluster'], cmap='rainbow')
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
plt.show()

# WCSSの値の確認
wcss_vals = []
for i in range(1, 16):
    kmeans_iter = KMeans(i)
    kmeans_iter.fit(target)
    wcss_vals.append(kmeans_iter.inertia_)

cluster_nums = range(1, 16)
plt.plot(cluster_nums, wcss_vals)
plt.title('The elbow methods')
plt.xlabel('cluster_nums')
plt.ylabel('wcss_vals')
plt.show()