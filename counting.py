import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import math
import time
import onnx
import onnxruntime as ort
import calculator

# https://arxiv.org/pdf/1312.7560.pdf
# file:///Users/cristiangutierrez/Downloads/1-s2.0-S187705092031526X-main.pdf

# TODO : Reducir el 'blinking' con una cola de los ultimos 10 frames
# TODO : Evitar sombras (DILATE O OPEN)
# TODO : 5s con num, 5s con simbolo

def getSignMapping(letter):
    if str(letter) == "1":
        return "+"
    if str(letter) == "2":
        return "-"
    if str(letter) == "3":
        return "*"
    if str(letter) == "4":
        return "/"
    if str(letter) == "0":
        return "="
    else:
        return "None"

def center_crop(frame):
    h, w = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]
    


if __name__ == '__main__':

    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    cap = cv2.VideoCapture(0)
    counting = True

    prev_n = ""
    output = ""

    while True:
        t_end = time.time() + 5
        while time.time() < t_end:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            height, width, channels = img.shape

            hand_zone = img[0:height - round(height/3), width - round(width/3) - ((height - round(height/3) + (width - width - round(width/3)))):width]

            gray = cv2.cvtColor(hand_zone, cv2.COLOR_BGR2GRAY)

            if counting:
                color = (50,205,50)
            else:
                color = (0, 0, 255)
            
            # Reduir soroll webcam
            gaus = cv2.GaussianBlur(gray, (5, 5), 0)

            # Calculem el otsu thresholding value (amb l'ajuda del histograma...)
            otsu_threshold, bin_im = cv2.threshold(
                gaus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            # Fem opening per desfernos de inner zones
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 7))
            # dilate = cv2.morphologyEx(bin_im, cv2.MORPH_DILATE, kernel)
            opening = cv2.morphologyEx(bin_im, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            
            if len(contours) != 0:
                segmented = max(contours, key=cv2.contourArea)
                cv2.drawContours(hand_zone, [segmented], 0, (50,205,50), 4)
                convexHull = cv2.convexHull(segmented)
                cv2.drawContours(hand_zone, [convexHull], -1, (255, 0, 0), 2)
                defects = cv2.convexityDefects(segmented, cv2.convexHull(segmented, returnPoints=False))
                
                areahull = cv2.contourArea(convexHull)
                areacnt = cv2.contourArea(segmented)
            
                arearatio=((areahull-areacnt)/areacnt)*100

                n=0
                
                defects_valids = []
                if defects is not None:
                    l=0
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(segmented[s][0])
                        end = tuple(segmented[e][0])
                        far = tuple(segmented[f][0])

                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        s = (a+b+c)/2
                        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                        
                        #distance between point and convex hull
                        d=(2*ar)/a
                        
                        # apply cosine rule here
                        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                        if angle <= 90 and d > 30:
                            l += 1
                            defects_valids.append(i)
                            cv2.circle(hand_zone, far, 3, [0,0,255], -1)
                        
                        if arearatio < 18:
                            n = 0
                        else:
                            n = len(defects_valids) + 1
                        
            # else:
                
            #     gray = cv2.flip(gray, 1)
            #     gray = center_crop(gray)
            #     color = (0, 0, 255)
            #     x = cv2.resize(gray, (28, 28))
            #     x = (x - mean) / std

            #     x = x.reshape(1, 1, 28, 28).astype(np.float32)
            #     y = ort_session.run(None, {'input': x})[0]

            #     index = np.argmax(y, axis=1)
            #     n = index_to_letter[int(index)]
            #     sign = getSignMapping(n)
            #     n = str(n) + " (" + sign + ")"

            if not counting:
                sign = getSignMapping(n)
                show = output + str(n) + "(" + sign + ")"
            else:
                show = output + str(n)
            
            cv2.rectangle(img, (width - round(width/3) - ((height - round(height/3) + (width - width - round(width/3)))), 0), (width-7,height - round(height/3)-7), color, 10)
            cv2.putText(img,str(show), (width - round(width/3) - 40,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
            
            # SHOW HISTOGRAM OF PICTURE
            # plt.hist(img[0:height - round(height/3), width - round(width/3) - ((height - round(height/3) + (width - width - round(width/3)))):width].ravel(), 256, [0, 256])
            # plt.draw()
            # plt.pause(0.1)
            # plt.clf()

            cv2.imshow("Image", hand_zone)
            cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)


            # cv2.imshow("Open", opening)
            # cv2.setWindowProperty("Open", cv2.WND_PROP_TOPMOST, 1)
            # cv2.waitKey(1)

            # cv2.imshow("Normal", bin_im)
            # cv2.setWindowProperty("Normal", cv2.WND_PROP_TOPMOST, 1)
            # cv2.waitKey(1)

            # cv2.imshow("Dilate", dilate)
            # cv2.setWindowProperty("Dilate", cv2.WND_PROP_TOPMOST, 1)
            # cv2.waitKey(1)


        if not counting:
            sign = getSignMapping(str(n))
            output = str(prev_n) + str(sign) + " "
            if calculator.checkEqual(str(n)):
                output = calculator.makeOperation(str(output))
                counting = not counting
        else:
            output = str(prev_n) + str(n) + " "

        prev_n = output

        counting = not counting
