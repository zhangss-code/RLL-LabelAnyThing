import numpy as np
import cv2
# SIFT 特征提取
def sift_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# 示例：读取图像并提取 SIFT 特征
image = cv2.imread('src\img\captured_test.jpg', 0)
keypoints, descriptors = sift_keypoints(image)

cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT Keypoints', image)
cv2.waitKey(0)
