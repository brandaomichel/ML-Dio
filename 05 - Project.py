import numpy as np
import cv2 as cv

# Corrected variable name from face_clasifier to face_classifier
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 

image = cv.imread('imagem.jpg')
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(image_gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow('Faces', image)
cv.waitKey(0)
cv.destroyAllWindows()