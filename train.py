import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Dataset
train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# Load EfficientNet
model = torchvision.models.efficientnet_b0(pretrained=True)

# Change output layer
model.classifier[1] = nn.Linear(1280, 2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):

    model.train()

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# =========================
# ACCURACY CHECKING
# =========================

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print("Accuracy:", 100 * correct / total)

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "models/pneumonia_model.pth")