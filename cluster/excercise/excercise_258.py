from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

df   = pd.read_csv('./Countries-exercise.csv')
c_df = df.copy()  # 念のためデータフレームをコピー

# クラスタリング
x      = df.iloc[:, 1:2] # Longitude と Latitudeのみ抽出
kmeans = KMeans(4)
kmeans.fit(x)
cluster = kmeans.fit_predict(x)
# 書き出し
c_df['Cluster'] = cluster

# 描画
plt.scatter(c_df['Longitude'], c_df['Longitude'], c=c_df['Cluster'], cmap='rainbow')
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
plt.show()