import pandas as pd

"""
データの取得～確認まで
"""
df   = pd.read_csv('./Absenteeism-data.csv')
c_df = df.copy()
# print(df.info())
# print(df.describe())

"""
データ前処理
列削除・列追加・カテゴリ変数 -> ダミー変数・標準化など
"""
c_df = c_df.drop(['ID'], axis=1)

# 欠席理由コードをダミー変数化し、グループ化
reason_dummies = pd.get_dummies(c_df['Reason for Absence'], drop_first=True) # n-1個のdummy変数を作成
# print('reason col dummies', reason_dummies.columns.values) # いくつの変数が作られているか
# reason_dummies['checks'] = reason_dummies.sum(axis=1)      # 各行の最大値
# print('reason col checks', reason_dummies['checks'].unique()) # 1のみであればOK
# reason_dummies = reason_dummies.drop(['checks'], axis=1)
c_df           = c_df.drop(['Reason for Absence'], axis=1)
reason_group1  = reason_dummies.loc[:,  1:14].max(axis=1).map({False: 0, True: 1})
reason_group2  = reason_dummies.loc[:, 15:17].max(axis=1).map({False: 0, True: 1})
reason_group3  = reason_dummies.loc[:, 18:21].max(axis=1).map({False: 0, True: 1})
reason_group4  = reason_dummies.loc[:, 22:  ].max(axis=1).map({False: 0, True: 1})
c_df           = pd.concat([c_df, reason_group1, reason_group2, reason_group3, reason_group4], axis=1)
c_df.columns   = list(c_df.columns.values[:-4]) + ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']  # 結合した後ろ4つのカラム名を変更
df_checkpoint  = c_df.copy() # チェックポイント


# 日付データの整形
c_df['Date']            = pd.to_datetime(c_df['Date'], format='%d/%m/%Y')
c_df['Month Value']     = c_df['Date'].apply(lambda date_val: date_val.month)
c_df['Day of the Week'] = c_df['Date'].apply(lambda date_val: date_val.weekday())
c_df                    = c_df.drop(['Date'], axis=1)
columns_names_recorded  = [
    'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value' ,'Day of the Week', 'Transportation Expense' ,'Distance to Work',
    'Age', 'Daily Work Load Average', 'Body Mass Index' ,'Education', 'Children', 'Pets', 'Absenteeism Time in Hours'
]
# 整列
df_reason_date_mod      = c_df[columns_names_recorded]
df_checkpoint           = df_reason_date_mod.copy() # チェックポイント

# 学歴の整形（値:件数で表示 1:583, 2:40, 3:73, 4:4 なので、1とそれ以外で分ける）
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
df_remake_complete = df_reason_date_mod.copy()  # チェックポイントかつ、外部への露出用

# 念のため出力もしておきたい場合は、以下を利用
df_remake_complete.to_csv('./df_remake_complete.csv', index=False)

"""
以下、課題コード
"""
# # 年齢をダミー変数化
# age_dummies = pd.get_dummies(c_df['Age'], drop_first=True) # n-1個のdummy変数を作成
# print('age col checks', age_dummies.columns)
# age_dummies['checks'] = age_dummies.sum(axis=1)  # 確認のため、一度追加
# print('age col checks', age_dummies['checks'].unique())
# age_dummies = age_dummies.drop(['checks'], axis=1)
# df_no_age = c_df.drop(['Age'], axis=1)
# df_concatenated = pd.concat([df_no_age, age_dummies], axis=1)
# df_checkpoint = df_concatenated.copy() # チェックポイント

# # 整列
# columns_names_recorded_3 = [
#     'Date', 'Transportation Expense', 'Distance to Work' ,'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets' , 'Reason_1',
#     'Reason_2', 'Reason_3', 'Reason_4',28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 46, 47, 48, 49, 50, 58, 'Absenteeism Time in Hours',
# ]
# df_concatenated = df_concatenated[columns_names_recorded_3]
# df_checkpoint = df_concatenated.copy() # チェックポイント