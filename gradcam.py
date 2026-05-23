from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
import torch
import cv2
import numpy as np

model = torchvision.models.efficientnet_b0()

model.classifier[1] = torch.nn.Linear(1280, 2)

model.load_state_dict(torch.load("models/pneumonia_model.pth"))

target_layer = model.features[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

image = cv2.imread("sample.jpg")

image = cv2.resize(image, (224,224))

rgb_img = np.float32(image) / 255

input_tensor = torch.tensor(rgb_img).permute(2,0,1).unsqueeze(0).float()

grayscale_cam = cam(input_tensor=input_tensor)[0]

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

cv2.imwrite("heatmaps/output.jpg", visualization)