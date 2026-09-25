import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from data_preprocessing import word_to_idx, x_test_tensor, x_train_tensor, y_test_tensor, y_train_tensor

vocab_size = len(word_to_idx)

embedding_dim = 128
hidden_dim = 128
num_classes = 6

batch_size = 32
learning_rate = 0.0001
num_epochs = 10

x_train = x_train_tensor
y_train =y_train_tensor


print("\nTrain size:", len(x_train))
print("Test size:", len(x_test_tensor))

class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
print("\nClass counts:", class_counts)

class_weights = 1.0 / np.sqrt(class_counts ** 0.99)
class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float)

sample_weights = torch.tensor(class_weights[y_train.numpy()], dtype=torch.double)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print("\nClass weights:", class_weights)

class RNNclassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            nonlinearity="tanh"
            )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(), 
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
            )

    # def forward(self, x):
    #     padding_mask = (x != 0).float()
    #     x = self.embedding(x)
    #     rnn_out, _ = self.rnn(x)
    #     mask = padding_mask.unsqueeze(-1)
    #     masked_rnn_out = rnn_out * mask
    #     summed = masked_rnn_out.sum(dim=1)
    #     token_count = mask.sum(dim=1).clamp(min=1)
    #     out = summed / token_count
    #     return self.fc(out)

    def forward(self, x):
            padding_mask = (x != 0)
            x = self.embedding(x)
            lstm_out, _ = self.lstm(x)
    
            mask = padding_mask.unsqueeze(-1)
    
            masked_for_max = lstm_out.masked_fill(~mask, -1e9)
            max_pool = masked_for_max.max(dim=1).values
    
            return self.fc(max_pool)

model = RNNclassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_classes=num_classes)


criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1:02d}/{num_epochs}] Train Loss: {average_train_loss:.4f}")

model.eval()
all_predictions, all_labels = [], []
total_test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        total_test_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_loss = total_test_loss / len(test_loader)
test_accuracy = accuracy_score(all_labels, all_predictions)
test_macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
test_weighted_f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")

class_names = ["Safe", "Violent Crimes", "Non-Violent Crimes", "unsafe", "Unknown S-Type", "Sex-Related Crimes"]

print(confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes))))

print(classification_report(all_labels, all_predictions, labels=list(range(num_classes)), target_names=class_names, zero_division=0))