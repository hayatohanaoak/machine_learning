import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 初期値用意
obs = 1000
xs = np.random.uniform(10, -10, (obs, 1))
zs = np.random.uniform(10, -10, (obs, 1))
gen_inputs = np.column_stack((xs, zs))
# ターゲットデータの用意
noise = np.random.uniform(-1, 1, (obs, 1))
gen_targets = 2 * xs - 3 * zs + 5 + noise

# ファイルに書き出し
np.savez(
    './TF_intro',         # ファイル名
    inputs=gen_inputs,    # 入力値（inputsという名前である必要はない）
    targets=gen_targets   # 出力値（targetsという名前である必要はない）
)

# ファイルの読み込み
train_data = np.load('./TF_intro.npz')
input_size = 2
output_size = 1

# モデルの作成
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        output_size,
        kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),   # 重みの初期値（無くても良い）
        bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)      # バイアスの初期値（無くても良い）
    )
])
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)  # 学習率の指定（無くても良い）
model.compile(
    # optimizer='sgd',          # 最適化アルゴリズムの指定（今回は、sgd=確率的勾配降下法）
    optimizer=custom_optimizer,
    loss='mean_squared_error' # 損失関数の指定（今回はl2ノルム）
)

# モデルの訓練
model.fit(
    train_data['inputs'],
    train_data['targets'],
    epochs=100,  # 繰り返し回数
    # verbose=2    #  0 = silent, 1 = progress bar, 2 = single line
    verbose=0    #  0 = silent, 1 = progress bar, 2 = single line
)

# 重みとバイアスの出力
weights = model.layers[0].get_weights()[0]
biases  = model.layers[0].get_weights()[1]

# 予測
preds = model.predict_on_batch(train_data['inputs'])

# 予測値と出力値の比較（描画されるグラフが横並びになっていることを確認）
plt.plot(np.squeeze(preds), np.squeeze(train_data['targets']))
plt.show()