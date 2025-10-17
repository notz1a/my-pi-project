from transformers import pipeline
import cv2
from PIL import Image
import numpy as np

print("♻️ Loading Waste Classification Model...")
pipe = pipeline("image-classification", model="watersplash/waste-classification")
print("✅ Model ready!\n")

# mapping labels to simpler waste groups
WASTE_MAP = {
    "cardboard": "Recyclable - Paper/Cardboard",
    "glass": "Recyclable - Glass",
    "metal": "Recyclable - Metal",
    "paper": "Recyclable - Paper",
    "plastic": "Recyclable - Plastic",
    "trash": "Non-recyclable Waste",
}

def classify_frame(frame):
    """Capture frame → classify → return label + confidence"""
    # Convert OpenCV BGR → RGB for Pillow
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preds = pipe(image)
    label = preds[0]['label'].lower()
    conf = preds[0]['score']

    # group to waste type
    waste_type = WASTE_MAP.get(label, label.capitalize())
    return waste_type, conf
