import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import onnx
import onnxruntime as ort

# https://arxiv.org/pdf/1312.7560.pdf
# file:///Users/cristiangutierrez/Downloads/1-s2.0-S187705092031526X-main.pdf


if __name__ == '__main__':

    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    cap = cv2.VideoCapture(0)
    
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        height, width, channels = img.shape

        hand_zone = img[0:height - round(height/3), width - round(width/3) - ((height - round(height/3) + (width - width - round(width/3)))):width]

        gray = cv2.cvtColor(hand_zone, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(gray, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        cv2.rectangle(img, (width - round(width/3) - ((height - round(height/3) + (width - width - round(width/3)))), 0), (width,height - round(height/3)), (50,205,50), 4)
        
        cv2.putText(img,str(letter), (width - round(width/3) - 40,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Image", hand_zone)
        cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
