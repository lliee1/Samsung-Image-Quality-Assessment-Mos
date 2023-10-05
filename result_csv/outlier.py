import pandas as pd
from collections import defaultdict

test_640_fold_0 = pd.read_csv("./640_fold0.csv")
test_640_fold_1 = pd.read_csv("./640_fold1.csv")
test_640_fold_2 = pd.read_csv("./640_fold2.csv")
test_640_fold_3 = pd.read_csv("./640_fold3.csv")
test_640_fold_4 = pd.read_csv("./640_fold4.csv")

test_448_fold_0 = pd.read_csv("./448_fold0.csv")
test_448_fold_1 = pd.read_csv("./448_fold1.csv")
test_448_fold_2 = pd.read_csv("./448_fold2.csv")
test_448_fold_3 = pd.read_csv("./448_fold3.csv")
test_448_fold_4 = pd.read_csv("./448_fold4.csv")

test_384_fold_0 = pd.read_csv("./384_fold0.csv")
test_384_fold_1 = pd.read_csv("./384_fold1.csv")
test_384_fold_2 = pd.read_csv("./384_fold2.csv")
test_384_fold_3 = pd.read_csv("./384_fold3.csv")
test_384_fold_4 = pd.read_csv("./384_fold4.csv")


datas = [test_640_fold_0,test_640_fold_1,test_640_fold_2,test_640_fold_3,test_640_fold_4,
        test_448_fold_0,test_448_fold_1,test_448_fold_2,test_448_fold_3,test_448_fold_4,
        test_384_fold_0,test_384_fold_1,test_384_fold_2,test_384_fold_3,test_384_fold_4]

data = pd.concat(datas)


df = defaultdict(list)
for img in data.img_name.unique():
    d = data[data.img_name==img]
    Q1 = d.mos.quantile(.25)
    Q3 = d.mos.quantile(.75)
    IQR = Q3 - Q1
    mos = d.mos[(d.mos < Q3 + 1.5 * IQR) & (d.mos > Q1 - 1.5 * IQR)].mean()
    
    df["img_name"] += [img]
    df["mos"] += [mos]
    df["commnets"] += [""]

pd.DataFrame(df).to_csv("./final.csv",index=False)
    