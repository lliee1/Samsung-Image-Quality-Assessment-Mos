import pandas as pd


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

path = test_640_fold_0["img_name"]
cap = test_640_fold_0["comments"]


last_csv = pd.DataFrame(path, columns=["img_name"])
last_csv["mos"] = (
    test_640_fold_0["mos"]
    + test_640_fold_1["mos"]
    + test_640_fold_2["mos"]
    + test_640_fold_3["mos"]
    + test_640_fold_4["mos"]
    + test_448_fold_0["mos"]
    + test_448_fold_1["mos"]
    + test_448_fold_2["mos"]
    + test_448_fold_3["mos"]
    + test_448_fold_4["mos"]
    + test_384_fold_0["mos"]
    + test_384_fold_1["mos"]
    + test_384_fold_2["mos"]
    + test_384_fold_3["mos"]
    + test_384_fold_4["mos"]
) / 15
last_csv["comments"] = cap

last_csv.to_csv(
    "./15fold_final.csv",
    mode="w",
    index=False,
)
