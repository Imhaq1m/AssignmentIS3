import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --------------------------
# 1. Dataset Preparation
# --------------------------
# Download dataset

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize for grayscale converted to RGB
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageFolder(root='dataset/train', transform=transform)
test_dataset = ImageFolder(root='dataset/test', transform=transform)

# Optional: Split training into train + validation
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Classes:", train_dataset.classes)

# --------------------------
# 2. Model Creation
# --------------------------


def get_shufflenet_model(num_classes):
    # You can use shufflenet_v2_x0_5 for smaller size
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)
model = get_shufflenet_model(num_classes).to(device)

# Print model summary
print(model)

# --------------------------
# 3. Training the Model
# --------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation step
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        print(f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_shufflenet_emotion.pth')

    print('Training complete.')
    return model


# Start training
model = train_model(model, num_epochs=15)

# --------------------------
# 4. Evaluation
# --------------------------


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")


evaluate_model(model, test_loader)


def visualize_predictions(model, data_loader, classes):
    model.eval()
    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 6))
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        plt.imshow(img[..., 0], cmap='gray')
        plt.title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


visualize_predictions(model, test_loader, train_dataset.classes)
