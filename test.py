import pandas as pd
import random


df = pd.read_csv('/root/dacon/data/submit_maniqa.csv' )
# print(df)
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(len(df))

# print(df.info())
# print(df.dtypes)
# print(df.columns)
# print(df.describe())
ranges = 6.719480519479999
min = 1.88051948052


img_name_ls = []
train_mos_ls = []
nice_ls = []

for mos in df['mos']:
    mos = round(mos*ranges+min,11)
    if mos >10:
        mos -= 2
    train_mos_ls.append(mos)

for img_path in df['img_name']:
    img_name_ls.append(img_path)

for _ in range(13012):
    nice_ls.append("Nice image")

train_df = pd.DataFrame(img_name_ls, columns=['img_name'])
train_df.insert(1, 'mos', train_mos_ls)
train_df.insert(2, 'comments', nice_ls)
train_df.to_csv('/root/dacon/data/submit_maniqa_edit.csv', mode='w', index=False)

# print(len(df)/4)



# temp_list = [x for x in range(74568)]
# temp = random.sample(temp_list, 18642)

# val_data_path = []
# val_mos_ls = []
# for idx in temp:
#     mos_temp = train_mos_ls[idx]
#     data_path_temp = train_data_path[idx]
#     val_data_path.append(data_path_temp)
#     val_mos_ls.append(mos_temp)
#     train_data_path[idx] = ''
#     train_mos_ls[idx] = ''

# train_data_path = [value for value in train_data_path if value != '']
# train_mos_ls = [value for value in train_mos_ls if value != '']

# train_df = pd.DataFrame(train_data_path, columns=['img_path'])
# train_df.insert(1, 'mos', train_mos_ls)

# val_df = pd.DataFrame(val_data_path, columns=['img_path'])
# val_df.insert(1, 'mos', val_mos_ls)

# train_df.to_csv('/root/dacon/data/train_df.csv', mode='w')
# val_df.to_csv('/root/dacon/data/val_df.csv', mode='w')