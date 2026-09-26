import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from collections import Counter
import re


df = pd.read_csv("train_true_labels.csv")

print(df.head())
print(df.shape)

print(df.columns)

print(df.isnull().sum())

# print(list(df["Toxic Category"]))

print(df["query"].str.len().describe())

label_map = {
    "Safe": 0,
    "Violent Crimes": 1,
    "Non-Violent Crimes": 2,
    "unsafe": 3,
    "Unknown S-Type": 4,
    "Sex-Related Crimes": 5,
    "Suicide & Self-Harm": 6,
    "Elections": 7,
    "Child Sexual Exploitation": 8
}


print("Shape before:", df.shape)

print("Number of duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates(subset=["query"])

print("Shape after:", df.shape)

print("Number of null rows:", df.isnull().sum())


df["label"] = df["True Label"].map(label_map)


df["text"] = (
    df["query"].fillna("") + " " +
    df["image descriptions"].fillna("")
)

x= df["text"]
y= df["label"]

mask = ~y.isin([7, 8])

x = x[mask]
y = y[mask]


print("Label distribution:")
print(y.value_counts().sort_index())

x_train ,x_test ,y_train ,y_test =train_test_split(x ,y ,test_size=0.2 ,shuffle=True,random_state=42,
    stratify=y)

# x_train = x_train.to_frame()

# from imblearn.over_sampling import RandomOverSampler
# from collections import Counter

# print("Before:")
# print(Counter(y_train))

# # Oversampling
# ros = RandomOverSampler(
#     sampling_strategy={
#         0: 24,   # unsafe
#         1: 24,   # Unknown S-Type
#         2: 30    # Sex-Related Crimes
#     },
#     random_state=42
# )

# if isinstance(x_train, pd.Series):
#     x_train = x_train.to_frame()

# x_train, y_train = ros.fit_resample(
#     x_train,
#     y_train
# )
# x_train = x_train.iloc[:, 0]

print(type(x_train))
print(x_train.head())

print("\nAfter:")
print(Counter(y_train))

print("Ddsvdsv",y_train.value_counts())


def tokenize(text):
    text = text.lower()    
    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b|[!?]", text)
    
    return tokens


# vocabulary
counter =Counter()
for text in x_train:
    counter.update(tokenize(text))
print("Unique words:", len(counter))


# token
word_to_idx ={
    "<PAD>":0,
    "<UNK>":1
}

for word,count in counter.most_common():
    word_to_idx[word] =len(word_to_idx)

print("total num of token",len(word_to_idx)) # 5021

# text -> seq ("hello world" -> [52, 73])
def text_to_sequence(text):
    tokens = tokenize(text)

    return [
        word_to_idx.get(word, word_to_idx["<UNK>"])
        for word in tokens
    ]

train_lengths = x_train.apply(lambda text: len(tokenize(text)))

max_len = int(train_lengths.quantile(0.99))
print("maxxxx len", max_len)

# padding 
def pad_seq(sequence ,max_len):
    if len(sequence)> max_len:
        return sequence[:max_len]
    return sequence + [word_to_idx["<PAD>"]] * (max_len -len(sequence))


x_train_seq =[
    pad_seq(text_to_sequence(text) ,max_len)
    for text in x_train
]
x_test_seq =[pad_seq(text_to_sequence(text),max_len) for text in x_test]

print(x_train_seq[:4])



x_train_tensor = torch.tensor(x_train_seq ,dtype=torch.long)
x_test_tensor =torch.tensor(x_test_seq ,dtype =torch.long)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.long)

print("Train:", x_train_tensor.shape)
print("Test:", x_test_tensor.shape)
print("Vocabulary:", len(word_to_idx))
print("Max length:", max_len)


