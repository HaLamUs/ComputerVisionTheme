import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread('hoailinh.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

id = 1
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cropped = img[ y : y+h, x : x+w ]
    cv.imwrite("cropped_face" + str(id) + ".png", cropped)
    id+=1

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()