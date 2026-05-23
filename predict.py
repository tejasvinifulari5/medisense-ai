import torch
import torchvision
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Load model
model = torchvision.models.efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = torch.nn.Linear(1280, 2)

model.load_state_dict(
    torch.load("models/pneumonia_model.pth")
)

model.eval()

# Load image
image = Image.open("sample.jpg").convert("RGB")

image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():

    output = model(image)

    _, prediction = torch.max(output, 1)

classes = ["NORMAL", "PNEUMONIA"]

print("Prediction:", classes[prediction.item()])