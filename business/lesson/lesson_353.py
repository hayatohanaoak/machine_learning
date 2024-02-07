import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# データを読み込んで、入力値と出力値に分割
raw_csv_data        = np.loadtxt('./Audiobooks-data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1] # 最初の列と最後の列を除去
targets_all         = raw_csv_data[:, -1]  # 最終列のみ


# バランシング
num_one_targets      = int(np.sum(targets_all))
zero_targets_counter = 0
indicates_to_remove  = []
# targetsの列で、 「値が1であるデータの数」と同じ回数だけの「値が0であるデータの数」が出現した行数を特定する
# そして、それ以降の行で、値が0となるデータの行数を抽出
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indicates_to_remove.append(i)  # 1の値と等しくなって以降の「値が0の列」は不要リストへ
unscaled_inputs_equal_piriors = np.delete(unscaled_inputs_all, indicates_to_remove, axis=0)
targets_equal_priors          = np.delete(targets_all, indicates_to_remove, axis=0)


# 入力値の標準化（データの尺度をそろえる）
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_piriors)


# データシャッフル（行数をシャッフルして、その行数をもとにデータを引っ張ってランダム化）
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)                # ここまでで行数のシャッフル
shuffled_inputs  = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


# データの数を、 トレーニング:検証:テスト = 8:1:1 になるようにデータ数を分割
samples_count       = shuffled_inputs.shape[0]
train_samples_count = int(0.7 * samples_count)
valid_samples_count = int(0.15 * samples_count)
test_samples_count  = samples_count - train_samples_count - valid_samples_count
# 入力データと出力データを実際に トレーニング:検証:テスト = 8:1:1 分ける
train_inputs  = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
valid_inputs  = shuffled_inputs[:valid_samples_count]
valid_targets = shuffled_targets[:valid_samples_count]
test_inputs   = shuffled_inputs[:test_samples_count]
test_targets  = shuffled_targets[:test_samples_count]
# 出力値の値が、50:50 になっているか確認
# print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
# print(np.sum(valid_targets), valid_samples_count, np.sum(valid_targets) / valid_samples_count)
# print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

# モデルの数値設定（入力値が748個、出力値は10個、隠れ層は50個）
INPUT_SIZE        = 10
OUTPUT_SIZE       = 2
HIDDEN_LAYER_SIZE = 50
# モデル作成
model = tf.keras.Sequential([
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax')
])
# 学習開始（early_stoppingで過学習を防止）
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2) # docs:https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_inputs,
    train_targets,
    batch_size=100,
    epochs=100,
    callbacks=[early_stopping,],
    validation_data=(valid_inputs, valid_targets),
    verbose=2
)

# モデルのテスト
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

print(f'test loss: {test_loss}, test accuracy: {test_accuracy*100}')