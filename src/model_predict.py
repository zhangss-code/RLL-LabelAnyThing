import cv2
import numpy as np
#灰度处理

gray_img = cv2.imread('src\img\captured_test.jpg')
gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray_img)
edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)
#边缘检测
cv2.imshow('Canny Edges', edges)

for i in range(3):
    gray_img = cv2.imread('train_data/')
cv2.waitKey(0)
