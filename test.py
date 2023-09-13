import pandas as pd

# train = pd.read_csv("./data/train.csv")

# caption = train.comments

# for comments in caption:
#     with open("./caption.txt", "a") as f:
#         a = comments + "\n"
#         f.write(a)

f = open("./caption.txt", "r")


wordCount = {}


lines = f.readlines()
for line in lines:
    wordList = line.split()
    for word in wordList:
        # Get 명령어를 통해, Dictionary에 Key가 없으면 0리턴

        wordCount[word] = wordCount.get(word, 0) + 1

keys = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
f.close()
print(keys)

with open("./statis.txt", "a") as f:
    for word in keys:
        f.write(str(word[0]) + ":" + str(word[1]) + "\n")
