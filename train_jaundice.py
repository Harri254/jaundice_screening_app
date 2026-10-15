import os
import time
import copy
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from sklearn.metrics import accuracy_score

# ==========================================
# 1. DATASET DEFINITIONS (GLOBAL SCOPE)
# ==========================================
class JaundiceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        # Load positive (jaundice) images -> Label 1
        pos_dir = os.path.join(data_dir, 'positive')
        if os.path.exists(pos_dir):
            for img_name in os.listdir(pos_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(pos_dir, img_name))
                    self.labels.append(1)

        # Load negative (normal) images -> Label 0
        neg_dir = os.path.join(data_dir, 'negative')
        if os.path.exists(neg_dir):
            for img_name in os.listdir(neg_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(neg_dir, img_name))
                    self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)


class TorchvisionDatasetWrapper(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = JaundiceDataset(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.dataset.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class JaundiceClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(JaundiceClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.squeeze(1)


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cpu'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    DATA_DIR = './dataset'
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    image_datasets = {
        'train': TorchvisionDatasetWrapper(os.path.join(DATA_DIR, 'train'), transform=train_transforms),
        'val': TorchvisionDatasetWrapper(os.path.join(DATA_DIR, 'val'), transform=val_transforms)
    }

    # Note: On Windows, num_workers=0 avoids multiprocessing spawning issues. 
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    model = JaundiceClassifier(model_name='efficientnet_b0', pretrained=True)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)

    torch.save(trained_model.state_dict(), 'jaundice_efficientnet_b0.pth')
    print("Model saved successfully to 'jaundice_efficientnet_b0.pth'")