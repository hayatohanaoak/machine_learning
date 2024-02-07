from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 訓練用データの作成  本来は外部から取得したデータを sklearn.model_selection の train_test_splitで作成する
obs    = 1000
xs     = np.random.uniform(low=-10, high=10, size=(obs, 1))  # 1000 * 1 の行列
zs     = np.random.uniform(-10, 10, (obs, 1))
inputs = np.column_stack((xs, zs))  # 2変数を結合（1000 * 2の行列）

# ターゲットの作成（正解となる値の作成） 今回は適当に作成…
noise   = np.random.uniform(-1, 1, (obs, 1))  # noiseは、線形性を保った形で定義しなければならない
targets = 2 * xs - 3 * zs + 5 + noise

# 訓練データのプロット（matplotlib.pyplotを使って3D散布図を作成）
plt_targets = targets.reshape(obs, 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, plt_targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=100)
plt.show()

# 変数の初期値設定（アルゴリズムの機能を保証するために設定する、十分に小さな値）
init_range    = 0.1  # ここの範囲は、十分に小さくする
learning_rate = 0.02 # 学習率の設定
biases        = np.random.uniform(-init_range, init_range, 1)
weights       = np.random.uniform(-init_range, init_range, (2, 1))

# モデルの作成
for i in range(200):
    outputs      = np.dot(inputs, weights) + biases  # 1000 * 1 行列
    deltas       = outputs - targets

    # 損失・バイアス・重みの計算
    loss         = np.sum(deltas **2) / 2 / obs  # L2ノルム（ユークリッド距離）を2とobsで割る
    delta_scaled = deltas / obs
    biases       = biases  - learning_rate * np.sum(delta_scaled)
    weights      = weights - learning_rate * np.dot(inputs.T, delta_scaled)
    """
        np.dot(inputs.T, delta_scaled)の「inputs.T」をする意味
        
        inputs       = 1000 * 2 行列
        delta_sacled = 1000 * 1 行列
        inputs の行数と、 delta_scaledの列数が揃っていないため、計算不可。

        よって、inputsの転置を行って、
        inputs       = 2 * 1000 行列
        delta_sacled = 1000 * 1 行列
        に変形し、inputs の行数とdelta_scaledの列数を揃える。
    """
    print('===============  ')
    print('loss=', loss)
    print('weights=', weights, 'biases=', biases)

# 最終出力（x軸）とターゲット（y軸）のグラフ
plt.plot(outputs, targets)
plt.show()
