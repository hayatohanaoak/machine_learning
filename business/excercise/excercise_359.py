# レッスンで作成したコードをベースに、インデックスをシャッフルした上でデータセットのバランシングをしてみましょう。

import numpy as np
from sklearn import preprocessing

# データを読み込んで、入力値と出力値に分割
raw_csv_data        = np.loadtxt('../lesson/Audiobooks-data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1] # 最初の列と最後の列を除去
targets_all         = raw_csv_data[:, -1]  # 最終列のみ

# データシャッフル（行数をシャッフルして、その行数をもとにデータを引っ張ってランダム化）
shuffled_indices            = np.arange(unscaled_inputs_all.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_unscaled_input_all = unscaled_inputs_all[shuffled_indices]
shuffled_targets_all        = targets_all[shuffled_indices]

# バランシング
num_one_targets      = int(np.sum(shuffled_targets_all))
zero_targets_counter = 0
indicates_to_remove  = []
# targetsの列で、 「値が1であるデータの数」と同じ回数だけの「値が0であるデータの数」が出現した行数を特定する
# そして、それ以降の行で、値が0となるデータの行数を抽出
for i in range(shuffled_targets_all.shape[0]):
    if shuffled_targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indicates_to_remove.append(i)  # 1の値と等しくなって以降の「値が0の列」は不要リストへ
shuffled_unscaled_inputs_equal_piriors = np.delete(shuffled_unscaled_input_all, indicates_to_remove, axis=0)
shuffled_targets_equal_priors          = np.delete(shuffled_targets_all, indicates_to_remove, axis=0)

# 入力値の標準化（データの尺度をそろえる）
scaled_inputs = preprocessing.scale(shuffled_unscaled_inputs_equal_piriors)

# データの数を、 トレーニング:検証:テスト = 8:1:1 になるようにデータ数を分割
samples_count       = scaled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
valid_samples_count = int(0.1 * samples_count)
test_samples_count  = samples_count - train_samples_count - valid_samples_count
# 入力データと出力データを実際に トレーニング:検証:テスト = 8:1:1 分ける
train_inputs  = shuffled_unscaled_inputs_equal_piriors[:train_samples_count]
train_targets = shuffled_targets_equal_priors[:train_samples_count]
valid_inputs  = shuffled_unscaled_inputs_equal_piriors[:valid_samples_count]
valid_targets = shuffled_targets_equal_priors[:valid_samples_count]
test_inputs   = shuffled_unscaled_inputs_equal_piriors[:test_samples_count]
test_targets  = shuffled_targets_equal_priors[:test_samples_count]
# 出力値の値が、50:50 になっているか確認
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(valid_targets), valid_samples_count, np.sum(valid_targets) / valid_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)