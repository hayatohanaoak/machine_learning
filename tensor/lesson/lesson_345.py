import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# 元データの作成
mnist_dataset, mnist_info = tfds.load(
    name='mnist',
    with_info=True,
    as_supervised=True  # 正解の値も入れる
)
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# テストデータと検証データを用意し、自然数変換
num_valid_samples = 0.1 * mnist_info.splits['train'].num_examples
num_valid_samples = tf.cast(num_valid_samples, tf.int64)
num_test_samples  = 0.1 * mnist_info.splits['test'].num_examples
num_test_samples  = tf.cast(num_valid_samples, tf.int64)

# scaleMuseを使ってデータの整形
def scale(image, label):
    image = tf.cast(image, tf.float32) # 整数変換
    image /= 255
    return image, label
scaled_train_and_valid_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# データのシャッフル（BUFFER_SIZEごとにシャッフルする事で、メモリの節約を行う）
BUFFER_SIZE = 10000
shuffled_train_and_valid_data = scaled_train_and_valid_data.shuffle(BUFFER_SIZE)

# 検証データ・テストデータを作成
valid_data = shuffled_train_and_valid_data.take(num_valid_samples)
train_data = shuffled_train_and_valid_data.skip(num_valid_samples)  # 検証データを取り除いた残り

# BATCH_SIZEごとにデータを分け、訓練・検証・テストデータを作成
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
valid_data = valid_data.batch(num_valid_samples) # フォワードプロパゲーションを行うので、データ数を絞る必要なし
test_data  = test_data.batch(num_test_samples)   # フォワードプロパゲーションを行うので、データ数を絞る必要なし

# 検証データを入力値と出力値に分割
valid_inputs, valid_targets = next(iter(valid_data))

# モデルの数値設定（入力値が748個、出力値は10個、隠れ層は50個）
input_size        = 748
output_size       = 10
hidden_layer_size = 50

# モデル作成
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),          # 28 × 28 × 1 の行列を 28 × 1 に整形（横並びに変形）
    tf.keras.layers.Dense(  # 各入力 × 各重み + 各バイアス の値を計算（1層目）
        hidden_layer_size,
        activation='relu'   # 活性化関数の指定
    ),
    tf.keras.layers.Dense(  # 各入力 × 各重み + 各バイアス の値を計算（2層目）
        hidden_layer_size,
        activation='relu'   # 活性化関数の指定
    ),
    tf.keras.layers.Dense(  # 出力層
        output_size,
        activation='softmax'# 活性化関数の指定
    )
])

# 最適化アルゴリズムと損失関数の決定
model.compile(
    optimizer='adam',                        # 最適化アルゴリズム
    loss='sparse_categorical_crossentropy',  # 損失関数
    metrics=['accuracy']
)

# 訓練
NUM_EPOCHS = 5
VALID_STEP = num_valid_samples
model.fit(
    train_data,
    epochs=NUM_EPOCHS, # 訓練回数
    validation_data=(  # 検証データを使って過学習を防止
        valid_inputs, valid_targets
    ),
    validation_steps=VALID_STEP, # 検証回数
    verbose=0
)

# モデルの検証
test_loss, test_accuracy = model.evaluate(test_data)

print(f'test loss: {test_loss}, test accuracy: {test_accuracy*100}')