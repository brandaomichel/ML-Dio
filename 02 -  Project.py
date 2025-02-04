import cv2
from cv2 import *
from google.colab.patches import cv2_imshow as showw

img = cv2.imread("C:/Users/Michel/Downloads/img.jpg")
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
showw(img)

binary = cv2.GaussianBlur(imggray, (5, 5), 0)
(T, bin) = cv2.threshold(binary, 160, 255, cv2.THRESH_BINARY)
showw(bin)