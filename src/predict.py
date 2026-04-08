import cv2
import numpy as np 
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("../models/model.h5")

def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("Image not found")
        return

    resized_img = cv2.resize(img, (224, 224))
    img = resized_img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)

    if pred[0][0] >= 0.5:
        print("PNEUMONIA")
    else:
        print("NORMAL")

predict_image("../test_image.jpg") 

