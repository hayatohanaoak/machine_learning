import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1, 101)
b = np.arange(501, 601)

# データを7:3に分ける
a_train, a_test, b_train, b_test = train_test_split(
    a, # 分割したいデータ
    b, # 分割したいデータ
    test_size = 0.3, # テストデータの割合
    shuffle=True,    # データのシャッフルを行うかどうか（場合に応じて使う）
    random_state=42  # ランダムに取得する際に、取得対象を固定する（場合に応じて使う）
)

print(a_train, a_test, b_train, b_test)