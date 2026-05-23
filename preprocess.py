import cv2
import os

IMG_SIZE = 224

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image