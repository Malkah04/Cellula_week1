# Multi-Class Text Classification using RNN and LSTM

## 📌 Project Overview

This project is a **multi-class text classification** system built using **RNN and LSTM** models with PyTorch.

The project takes text queries as input and classifies them into one of **9 different categories**.

The complete NLP pipeline was built without using any pretrained NLP model or pretrained word embeddings. The text was processed from the raw data until it was converted into sequences that could be used by the neural networks.

The main pipeline is:

```text
Raw Text
   ↓
Data Cleaning & Label Correction
   ↓
Class Distribution Analysis
   ↓
Manual Minority-Class Augmentation
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Vocabulary Building
   ↓
Text-to-Sequence
   ↓
Padding
   ↓
Embedding
   ↓
RNN / LSTM
   ↓
Multi-Class Prediction
```

---

## 🎯 Project Objective

The main goal of this project is to understand and implement a complete text classification pipeline using recurrent neural networks.

Instead of using pretrained models such as BERT or pretrained word embeddings, the model learns the required representations directly from the training data.

The project compares two recurrent architectures:

* RNN
* LSTM

---

## 📊 Dataset

The original dataset contains text queries with their corresponding safety-related categories.

The original label mapping contains **9 categories**:

| Label | Category                  | Used for Training |
| ----: | ------------------------- | :---------------: |
|     0 | Safe                      |         ✅         |
|     1 | Violent Crimes            |         ✅         |
|     2 | Non-Violent Crimes        |         ✅         |
|     3 | unsafe                    |         ✅         |
|     4 | Unknown S-Type            |         ✅         |
|     5 | Sex-Related Crimes        |         ✅         |
|     6 | Suicide & Self-Harm       |         ✅         |
|     7 | Elections                 |         ❌         |
|     8 | Child Sexual Exploitation |         ❌         |

Although the original dataset contains 9 categories, the models in this project were trained on **7 classes**.

The `Elections` and `Child Sexual Exploitation` categories were not included in the training data.

### Training Classes

The actual classification task used during training was:

| Label | Category            |
| ----: | ------------------- |
|     0 | Safe                |
|     1 | Violent Crimes      |
|     2 | Non-Violent Crimes  |
|     3 | unsafe              |
|     4 | Unknown S-Type      |
|     5 | Sex-Related Crimes  |
|     6 | Suicide & Self-Harm |

Therefore, the final neural network output contains **7 classes**.

```python
num_classes = 7
```

The original label mapping was:

```python
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
```

The last two categories were excluded before the training stage.


---

## 🧹 Data Preparation

Before training the models, the original dataset went through several preparation steps.

### 1. Label Correction

During the initial data inspection, some text samples were found to have incorrect labels.

The labels were reviewed and corrected, and a new CSV file was created:

```text
train_true_labels.csv
```

Claude was used as an assisting tool during the label-review process.

The corrected labels were then used as the basis for the following preprocessing and training steps.

---

### 2. Class Distribution Analysis

After correcting the labels, the class distribution was checked.

Some categories contained very few samples. In some cases, a class contained only **one record**.

With such a small number of examples, the model would have very limited information from which to learn the characteristics of these classes.

---

### 3. Manual Minority-Class Augmentation

To improve the representation of extremely small classes, additional text records were manually added to the minority classes.

These new records were added to:

```text
train_true_labels.csv
```

This was done to provide the model with more examples of the underrepresented categories during training.

This step is considered **manual data augmentation for minority classes**, rather than simply duplicating existing records.

---

## 🔄 Text Preprocessing

After preparing the dataset, the text was converted into numerical representations that could be processed by the neural networks.

### 1. Text Cleaning

The text was cleaned before tokenization to remove unnecessary patterns and improve the quality of the input.

### 2. Tokenization

A custom tokenization process was used to split each text into individual tokens.

For example:

```text
"this is a test"
```

becomes:

```text
["this", "is", "a", "test"]
```

### 3. Vocabulary Building

A vocabulary was created from the training data.

Each word was assigned a unique integer index.

Special tokens were also included:

```text
<PAD>
<UNK>
```

`<PAD>` is used for padding shorter sequences, while `<UNK>` represents words that are not found in the vocabulary.

### 4. Text-to-Sequence

Each tokenized text was converted into a sequence of integer IDs.

For example:

```text
["this", "is", "a", "test"]
```

could become:

```text
[15, 8, 42, 91]
```

### 5. Padding

Since text samples have different lengths, padding was applied so that all sequences have the same length.

For example:

```text
[15, 8, 42]
[15, 8, 42, 91, 13]
```

becomes:

```text
[15, 8, 42, 0, 0]
[15, 8, 42, 91, 13]
```

This allows the sequences to be processed in batches by the neural network.

---

## 🧠 Model Architecture

Two different recurrent neural network architectures were implemented.

### RNN

The first model uses a standard Recurrent Neural Network.

The general architecture is:

```text
Input Sequence
      ↓
Embedding
      ↓
RNN
      ↓
Fully Connected Layer
      ↓
9-Class Output
```

The RNN processes the input sequence step by step while maintaining a hidden state that carries information from previous tokens.

---

### LSTM

The second model uses a Long Short-Term Memory network.

The general architecture is:

```text
Input Sequence
      ↓
Embedding
      ↓
LSTM
      ↓
Fully Connected Layer
      ↓
9-Class Output
```

LSTM introduces gates that help the network keep or forget information from previous time steps.

This makes it more suitable for learning longer dependencies in sequential text compared with a basic RNN.

---

## ⚙️ Training

The models were implemented and trained using **PyTorch**.

The training process includes:

1. Loading the prepared dataset
2. Preprocessing the text
3. Converting text into numerical sequences
4. Creating PyTorch tensors
5. Creating training and testing datasets
6. Loading the data using DataLoaders
7. Forward propagation
8. Calculating the loss
9. Backpropagation
10. Updating model parameters
11. Evaluating the trained model

---

## ⚖️ Class Imbalance

Class imbalance was one of the main challenges in this project.

Some categories had significantly fewer samples than others, including classes with extremely small numbers of records.

Several techniques were explored during the project to improve the model's ability to learn minority classes, including:

* Manual minority-class augmentation
* Weighted sampling
* Class weighting
* Oversampling
* Different RNN/LSTM architectures
* Different preprocessing approaches

The performance was evaluated using metrics that provide more information than accuracy alone.

---

## 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Confusion Matrix
* Classification Report

### Why Macro F1?

Because the dataset is imbalanced, Macro F1-score is useful for evaluating performance across all classes.

It calculates the F1-score for each class and then gives each class equal importance.

This makes it easier to identify whether the model is performing well only on majority classes or also learning minority classes.

---

## 📊 Results

### RNN

| Metric      | Result |
| ----------- | -----: |
| Accuracy    |    0.94 |
| Macro F1    |    0.84 |
| Weighted F1 |    0.94 |

### LSTM

| Metric      | Result |
| ----------- | -----: |
| Accuracy    |    0.99 |
| Macro F1    |    0.93 |
| Weighted F1 |    0.98 |

Detailed classification reports and confusion matrices will be included with the final experiment results.

---


## 🛠️ Technologies

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn

---

## 📁 Project Structure

```text
project/
│
├── train_true_labels.csv
│
├──  data_preprocessing.py
│
├── RNN/
│   └── rnn.py
│
├── LSTM/
│   └── train_lstm.py
│
├── requirements.txt
└── README.md
└── LICENSE
```

> The exact project structure may vary depending on the final organization of the repository.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Malkah04/Cellula_week1.git
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the preprocessing

```bash
python data_preprocessing.py
```

### 5. Train the RNN

```bash
python -m RNN.rnn
```

### 6. Train the LSTM

```bash
python -m LSTM.train_lstm
```

---

## 💡 Key Takeaway

This project demonstrates a complete **multi-class text classification pipeline built from scratch using RNN and LSTM**.

Starting from raw and initially noisy labels, the data was reviewed, corrected, and prepared for training. Extremely small classes were also augmented manually to improve their representation.

The text was then transformed into numerical sequences through custom tokenization, vocabulary construction, text-to-sequence conversion, and padding.

Finally, RNN and LSTM models were trained using PyTorch and evaluated using multiple classification metrics, with particular attention to performance across imbalanced classes.
