import cv2
from model_loader import ThyroidModel
import numpy as np
import os
from PIL import Image

model = ThyroidModel("model.json", 'thyroid_vgg16_model.h5')

def process_image(filepath):
    # Load image using OpenCV
    img = cv2.imread(filepath)
    if img is None:
        return None, 0.0
    
    # Preprocess for VGG16 (224x224)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    
    # Predict
    pred_label, prob = model.predict_nodule(img_normalized[np.newaxis, :, :, :])
    
    # Return result
    if pred_label == 'Malignant':
        confidence = round(prob[0][0] * 100, 2)
    else:
        confidence = round((1 - prob[0][0]) * 100, 2)
        
    return pred_label, confidence