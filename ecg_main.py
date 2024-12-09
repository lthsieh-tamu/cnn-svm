#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


# In[2]:


class ECGDataset(Dataset):
    """Custom dataset for loading ECG .MAT files, one file per ECG signal."""
    def __init__(self, base_path):
        self.data = []
        self.labels = []
        self.label_map = {'H-H': 0, 'H-L': 1, 'L-H': 2, 'L-L': 3}  # Folder to label mapping
        
        for folder_name, label in self.label_map.items():
            folder_path = os.path.join(base_path, folder_name)
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(folder_path, file_name)
                    mat_data = scipy.io.loadmat(file_path)

                    # Extract ECG signal from the .MAT file
                    ecg_signal = None
                    for key in mat_data.keys():
                        if key.startswith('__'):  # Skip metadata keys
                            continue
                        if isinstance(mat_data[key], np.ndarray):
                            ecg_signal = mat_data[key].flatten()
                            break
                    
                    if ecg_signal is None:
                        raise KeyError(f"No valid ECG data found in file: {file_path}")
                    
                    # Normalize and pad/trim signal
                    ecg_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
                    if len(ecg_signal) > 128:
                        ecg_signal = ecg_signal[:128]
                    elif len(ecg_signal) < 128:
                        ecg_signal = np.pad(ecg_signal, (0, 128 - len(ecg_signal)), mode='constant')
                    
                    self.data.append(ecg_signal)
                    self.labels.append(label)
        
        self.data = np.array(self.data).astype(np.float32)
        self.labels = np.array(self.labels).astype(np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x).unsqueeze(0), torch.tensor(y)


# In[3]:


# CNN Model
class CNNFeatureExtractor1D(nn.Module):
    """1D-CNN model for ECG feature extraction"""
    def __init__(self, input_length=128):
        super(CNNFeatureExtractor1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=5, padding=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=3, padding=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate flattened size dynamically
        dummy_input = torch.randn(1, 1, input_length)
        flattened_size = self._get_flattened_size(dummy_input)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 32)  # Feature vector size

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        features = torch.relu(self.fc2(x))  # Extracted features
        return features

    def _get_flattened_size(self, dummy_input):
        with torch.no_grad():
            x = self.pool1(torch.relu(self.conv1(dummy_input)))
            x = self.pool2(torch.relu(self.conv2(x)))
            return x.numel()


# In[4]:


# Training the CNN
def train_cnn(model, train_loader, num_epochs=100, learning_rate=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for ecg_signal, labels in progress_bar:
            ecg_signal, labels = ecg_signal.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model(ecg_signal)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": epoch_loss / len(train_loader)})
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[5]:


# Feature Extraction
def extract_features(loader, model):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for ecg_signal, label in loader:
            ecg_signal = ecg_signal.to(device)
            feature_vector = model(ecg_signal)
            features.append(feature_vector.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


# In[6]:


# Visualize Features
def visualize_features(features, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title("Feature Distribution (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Classes")
    plt.show()


# In[7]:


# SVM Training and Evaluation
def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels):
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
    svm_classifier.fit(train_features, train_labels)

    predictions = svm_classifier.predict(test_features)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["H-H", "H-L", "L-H", "L-L"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print(classification_report(test_labels, predictions))


# In[9]:


# Main
base_path = '/Users/lthsieh/TAMU/24Fall/ECEN649/Final_Codes/ecg_GT'
dataset = ECGDataset(base_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


cnn_model = CNNFeatureExtractor1D(input_length=128).to(device)

# Train CNN
train_cnn(cnn_model, train_loader)

# Extract Features
train_features, train_labels = extract_features(train_loader, cnn_model)
test_features, test_labels = extract_features(test_loader, cnn_model)

# Visualize Features
visualize_features(train_features, train_labels)

# Train and Evaluate SVM
train_and_evaluate_svm(train_features, train_labels, test_features, test_labels)

