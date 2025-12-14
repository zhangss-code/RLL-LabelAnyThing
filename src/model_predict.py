import cv2
import numpy as np
#灰度处理

# 读取图像并进行 Canny 边缘检测

for i in range(1,4):
    gray_img = cv2.imread('train_data/'+str(i)+'.png')
    if gray_img is None:
        print("图像读取失败，请检查图像路径。")
        continue
    
    #cv2.imshow('Original Image '+str(i), gray_img)
    
    gray_img = cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img,threshold1=50,threshold2= 150)


    #cv2.imshow('Canny Edges '+str(i), edges)
    #进行边缘点的排序前处理
    y_coords,x_coords = np.where(edges!=0)
    edges_points = np.array(list(zip(x_coords,y_coords)))
    print("图像 "+str(i)+" 中的边缘点坐标：")
    print(edges_points)

    #计算边缘点的梯度幅值
    sobel_x = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)


    # 按梯度幅值降序排序（强边缘在前）
    sorted_indices = np.argsort(-gradient_magnitude[y_coords, x_coords])
    sorted_by_mag = edges_points[sorted_indices]
    

    # 筛选前20%的强边缘点
    top_k = int(len(sorted_by_mag) * 0.618)
    strong_edges = sorted_by_mag[:top_k]
    #可视化强边缘点
    strong_edge_img = np.zeros_like(gray_img)
    for point in strong_edges:
        strong_edge_img[point[1], point[0]] = 255
    cv2.imshow('Strong Edges Image '+str(i), strong_edge_img)
    cv2.waitKey(0)
