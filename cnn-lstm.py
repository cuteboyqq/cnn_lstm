import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import os
import random


import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR


class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=32):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.labels = sorted(os.listdir(self.root_dir))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.data = self._load_data()  # Initialize the data attribute

    def _load_data(self):
        data = []
        labels = []
        for label in self.labels:
            label_path = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for seq_folder in os.listdir(label_path):
                seq_path = os.path.join(label_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue
                
                img_files = [f for f in sorted(os.listdir(seq_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # Ensure there are enough images to create a sequence
                if len(img_files) < self.sequence_length:
                    print(f"Not enough images to create a sequence in folder {seq_path}")
                    continue

                data.append((seq_path, img_files))
                labels.append(self.label_to_idx[label])

        return list(zip(data, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (seq_path, img_files), label = self.data[idx]

        # Check the type of img_files
        if not isinstance(img_files, list):
            raise TypeError(f"Expected list for img_files, got {type(img_files)}")

        # Randomly select a starting index for the sequence
        start_index = random.randint(0, len(img_files) - self.sequence_length)
        sequence_files = img_files[start_index:start_index + self.sequence_length]

        images = []
        for img_name in sequence_files:
            img_path = os.path.join(seq_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        return torch.stack(images), label
        # return torch.stack(images), self.data[idx][1]



# Define the CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the original classifier
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Pass each frame through ResNet
        cnn_features = torch.stack([self.resnet(x[:, t, :, :, :]) for t in range(seq_len)], dim=1)
        
        # Pass the features through LSTM
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use the output from the last time step
        last_hidden = lstm_out[:, -1, :]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Classify
        out = self.fc(last_hidden)
        
        return out

def compute_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    num_batches = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).mean(0)
        std += images.std(2).std(0)
        num_batches += 1
    mean /= num_batches
    std /= num_batches
    return mean, std

def main():
    # Parameters
    dataset_dir = 'data/image_data/'
    num_classes = 5
    batch_size = 8
    sequence_length = 24
    img_height = 224
    img_width = 224
    model_dir = 'models'  # Directory to save model checkpoints
    val_split = 0.2  # Fraction of data to use for validation
    num_epochs = 100
    patience = 30
    min_delta = 0

    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = ImageSequenceDataset(root_dir=dataset_dir, transform=transform, sequence_length=sequence_length)
    
    # Split dataset into train and validation sets
    num_samples = len(dataset)
    num_val_samples = int(num_samples * val_split)
    num_train_samples = num_samples - num_val_samples
    
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # In your optimizer setup:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Adding weight decay
    # Initialize the scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Track the best validation accuracy
    best_val_accuracy = 0.0
    
    best_val_loss = np.Inf
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequences, target_labels in train_loader:
            sequences, target_labels = sequences.to(device), target_labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequences.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for sequences, target_labels in val_loader:
                sequences, target_labels = sequences.to(device), target_labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, target_labels)
                val_loss += loss.item() * sequences.size(0)

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == target_labels).sum().item()
                total_samples += target_labels.size(0)
        
        avg_val_loss = val_loss / len(val_dataset)
        val_accuracy = 100 * correct_predictions / total_samples
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            checkpoint_path = os.path.join(model_dir, f'model_best.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved new best model checkpoint: {checkpoint_path}')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == '__main__':
    main()
