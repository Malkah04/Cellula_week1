import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from collections import Counter
import re
import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

df = pd.read_csv("train_true_labels.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
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
print("Number of duplicate rows:", df["query"].duplicated().sum())

df = df.drop_duplicates(subset=["query"])

print("Shape after:", df.shape)
print("Number of null rows:", df.isnull().sum())

df["label"] = df["True Label"].map(label_map)
df["text"] = df["query"].fillna("") + " " + df["image descriptions"].fillna("")

x = df["text"]
y = df["label"]

mask = ~y.isin([6, 7, 8])
x = x[mask]
y = y[mask]

print("Label distribution:")
print(y.value_counts().sort_index())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def augment_text(text):
    words = text.split()
    if len(words) < 3:
        return text
    random_words = list(set([w for w in words if w.isalnum()]))
    random.shuffle(random_words)
    for w in random_words:
        syns = get_synonyms(w)
        if syns:
            syn = random.choice(syns)
            return " ".join([syn if word == w else word for word in words])
    return text

minority_classes = [3, 4, 5]
target_count = 100

train_df = pd.DataFrame({"text": x_train, "label": y_train})
aug_rows = []

for lbl in minority_classes:
    cls_df = train_df[train_df["label"] == lbl]
    curr_cnt = len(cls_df)
    if 0 < curr_cnt < target_count:
        needed = target_count - curr_cnt
        for _ in range(needed):
            sample_txt = cls_df["text"].sample(1).values[0]
            aug_rows.append({"text": augment_text(sample_txt), "label": lbl})

if aug_rows:
    train_df = pd.concat([train_df, pd.DataFrame(aug_rows)], ignore_index=True)

x_train = train_df["text"]
y_train = train_df["label"]

print(type(x_train))
print(x_train.head())

print("\nAfter Augmentation:")
print(Counter(y_train))
print("Ddsvdsv", y_train.value_counts())

def tokenize(text):
    text = text.lower()    
    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b|[!?]", text)
    return tokens

counter = Counter()
for text in x_train:
    counter.update(tokenize(text))
print("Unique words:", len(counter))

word_to_idx = {"": 0, "": 1}
for word, count in counter.most_common():
    word_to_idx[word] = len(word_to_idx)

print("total num of token", len(word_to_idx))

def text_to_sequence(text):
    tokens = tokenize(text)
    return [word_to_idx.get(word, word_to_idx[""]) for word in tokens]

train_lengths = x_train.apply(lambda text: len(tokenize(text)))
max_len = int(train_lengths.quantile(0.99))
print("maxxxx len", max_len)

def pad_seq(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[:max_len]
    return sequence + [word_to_idx[""]] * (max_len - len(sequence))

x_train_seq = [pad_seq(text_to_sequence(text), max_len) for text in x_train]
x_test_seq = [pad_seq(text_to_sequence(text), max_len) for text in x_test]

print(x_train_seq[:4])

x_train_tensor = torch.tensor(x_train_seq, dtype=torch.long)
x_test_tensor = torch.tensor(x_test_seq, dtype=torch.long)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

print("Train:", x_train_tensor.shape)
print("Test:", x_test_tensor.shape)
print("Vocabulary:", len(word_to_idx))
print("Max length:", max_len)