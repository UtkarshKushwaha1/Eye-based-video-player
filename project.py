import cv2
import numpy as np
import vlc
import time

video = vlc.MediaPlayer("Shaun the Sheep.mp4");

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

coordinate_list=[None]
cap = cv2.VideoCapture(0)

while True:
    
    
    ret,frame = cap.read();

    face_rectangle = eye_cascade.detectMultiScale(frame,scaleFactor = 1.6,minNeighbors = 5)

    size = len(coordinate_list)

    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(255,255,0),7)
        coordinate_list.append(x)

    if size != len(coordinate_list):
        video.play()
    else:
        video.set_pause(1)
        
    cv2.imshow("Eyes Detector",frame)

    k = cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()