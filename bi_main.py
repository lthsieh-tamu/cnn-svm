#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter


# In[ ]:


# 1. Custom Dataset with Grouping for LOPO
class ECGDataset(Dataset):
    def __init__(self, folder_path, fixed_length=38400):
        """
        Initialize the dataset, ensuring all signals are of a fixed length.
        
        folder_path: Path to the base directory containing subfolders (V-H, V-L, A-H, A-L).
        fixed_length: Target length for all signals (pad or truncate to this length).
        """
        self.signals = []
        self.labels = []
        self.groups = []
        self.fixed_length = fixed_length  # Target length for padding/truncation
        label_map = {'V-H': 0, 'V-L': 2, 'A-H': 1, 'A-L': 3}  # Map folder names to labels
        
        participant_id = 0  # Simulate grouping by participant
        for label_name, label_value in label_map.items():
            folder = os.path.join(folder_path, label_name)
            for file in os.listdir(folder):
                if file.endswith('.mat'):
                    mat = sio.loadmat(os.path.join(folder, file))
                    signal = self.extract_signal(mat, file)
                    signal = self.adjust_length(signal)  # Pad or truncate to fixed length
                    self.signals.append(self.normalize(signal))  # Normalize the signal
                    self.labels.append(label_value)
                    self.groups.append(participant_id)
            participant_id += 1  # Increment participant ID for grouping

        # Debug: Print label distribution
        print("Label distribution:", Counter(self.labels))

    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        """
        Return the signal and corresponding label at the given index.
        """
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return signal, label
    
    @staticmethod
    def normalize(signal):
        """
        Normalize the ECG signal to have zero mean and unit variance.
        """
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    @staticmethod
    def extract_signal(mat, filename):
        """
        Dynamically extract ECG signal data from a .mat file and ensure it is 1D.
        
        mat: Loaded .mat file.
        filename: Name of the .mat file (for error reporting).
        
        Returns:
        signal: Extracted ECG signal as a 1D array.
        """
        # Attempt to find the largest numeric array in the .mat file
        possible_signals = {key: mat[key] for key in mat.keys() if isinstance(mat[key], np.ndarray)}
        if not possible_signals:
            raise ValueError(f"No valid numeric arrays found in {filename}. Please check the file structure.")

        # Return the array with the largest number of elements
        signal_key = max(possible_signals, key=lambda k: possible_signals[k].size)
        signal = possible_signals[signal_key]

        # Flatten 2D signals into 1D if necessary
        if signal.ndim > 1:
            signal = signal.flatten()

        return signal.squeeze()  # Ensure the signal is flattened

    def adjust_length(self, signal):
        """
        Pad or truncate the signal to match the fixed length.
        """
        if len(signal) > self.fixed_length:
            # Truncate if the signal is longer than the fixed length
            return signal[:self.fixed_length]
        elif len(signal) < self.fixed_length:
            # Pad with zeros if the signal is shorter than the fixed length
            padding = np.zeros(self.fixed_length - len(signal))
            return np.concatenate((signal, padding))
        return signal


# In[ ]:


# 2. Enhanced 1D-CNN Architecture for ECG Signal Feature Extraction
class Modified1DCNN(nn.Module):
    def __init__(self, input_size):
        super(Modified1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2)
        )

        # Automatically compute the input size for the first fully connected layer
        dummy_input = torch.zeros(1, 1, input_size)  # Batch size=1, channel=1, length=input_size
        with torch.no_grad():
            conv_output = self.conv_layers(dummy_input)
            print(f"Output shape after convolutional layers: {conv_output.shape}")
            conv_output_size = conv_output.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# In[ ]:


# 3. LOPO Cross-Validation with CNN for Feature Extraction and SVM Classification
def train_and_evaluate(model, dataset, groups, epochs=20, learning_rate=1e-4):
    logo = LeaveOneGroupOut()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    all_test_labels_valence = []
    all_predictions_valence = []
    all_test_labels_arousal = []
    all_predictions_arousal = []

    class_counts = Counter(dataset.labels)
    class_weights = {k: 1.0 / v for k, v in class_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.labels]

    patience = 5  # Early stopping patience

    for train_idx, test_idx in logo.split(dataset.signals, dataset.labels, groups):
        # Prepare train and test subsets
        train_data = torch.utils.data.Subset(dataset, train_idx)
        test_data = torch.utils.data.Subset(dataset, test_idx)
        sampler = WeightedRandomSampler([sample_weights[i] for i in train_idx], len(train_idx))
        train_loader = DataLoader(train_data, batch_size=8, sampler=sampler)
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

        # Train the CNN for feature extraction
        model.train()
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                signals, labels = signals.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))
            scheduler.step()
            print(f"Epoch {epoch+1} Loss: {train_losses[-1]:.4f}")

            # Early stopping
            if train_losses[-1] < best_loss:
                best_loss = train_losses[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Extract features using the trained CNN
        model.eval()
        train_features, train_labels = [], []
        test_features, test_labels = [], []
        
        with torch.no_grad():
            for signals, labels in train_loader:
                signals = signals.to(device)
                features = model(signals)
                train_features.append(features.cpu().numpy())
                train_labels.append(labels.numpy())
            
            for signals, labels in test_loader:
                signals = signals.to(device)
                features = model(signals)
                test_features.append(features.cpu().numpy())
                test_labels.append(labels.numpy())
        
        train_features = np.vstack(train_features)
        train_labels = np.hstack(train_labels)
        test_features = np.vstack(test_features)
        test_labels = np.hstack(test_labels)

        # Split labels into valence and arousal
        train_labels_valence = np.where(train_labels < 2, train_labels, train_labels - 2)
        test_labels_valence = np.where(test_labels < 2, test_labels, test_labels - 2)

        train_labels_arousal = np.where(train_labels % 2 == 0, train_labels // 2, train_labels // 2)
        test_labels_arousal = np.where(test_labels % 2 == 0, test_labels // 2, test_labels // 2)

        # Train SVM on extracted features for Valence
        svm_valence = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_valence.fit(train_features, train_labels_valence)
        y_pred_valence = svm_valence.predict(test_features)
        all_test_labels_valence.extend(test_labels_valence)
        all_predictions_valence.extend(y_pred_valence)

        # Train SVM on extracted features for Arousal
        svm_arousal = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_arousal.fit(train_features, train_labels_arousal)
        y_pred_arousal = svm_arousal.predict(test_features)
        all_test_labels_arousal.extend(test_labels_arousal)
        all_predictions_arousal.extend(y_pred_arousal)

    # Evaluate Valence
    print("Valence Classification Report:")
    print(classification_report(all_test_labels_valence, all_predictions_valence, target_names=['L', 'H']))
    cm_valence = confusion_matrix(all_test_labels_valence, all_predictions_valence)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_valence, annot=True, fmt='d', cmap='Blues', xticklabels=['L', 'H'], yticklabels=['L', 'H'])
    plt.title('Confusion Matrix - Valence')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Evaluate Arousal
    print("Arousal Classification Report:")
    print(classification_report(all_test_labels_arousal, all_predictions_arousal, target_names=['L', 'H']))
    cm_arousal = confusion_matrix(all_test_labels_arousal, all_predictions_arousal)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_arousal, annot=True, fmt='d', cmap='Blues', xticklabels=['L', 'H'], yticklabels=['L', 'H'])
    plt.title('Confusion Matrix - Arousal')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[ ]:


# Main Execution
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    data_path = '/Users/lthsieh/TAMU/24Fall/ECEN649/Final_Codes/bi_GT'
    dataset = ECGDataset(data_path)
    groups = dataset.groups

    # Define the CNN model
    signal_length = len(dataset[0][0].squeeze())
    model = Modified1DCNN(signal_length).to(device)

    # Train and evaluate using LOPO with SVM
    train_and_evaluate(model, dataset, groups, epochs=20, learning_rate=1e-4)

