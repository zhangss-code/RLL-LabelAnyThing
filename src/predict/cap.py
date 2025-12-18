import cv2
import numpy as np
def A_CAM_P():
    
    Open_camrea = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not Open_camrea.isOpened():
        print("摄像头可能出现问题，请检查摄像头连接。")

    jpg = Open_camrea.read()[1]
    if jpg is None:
        print("摄像头读取失败，请检查摄像头连接。")
        return None
    
    print("摄像头读取成功。")
    # 显示图像
    cv2.imshow("Camera", jpg)
    cv2.imwrite("src\img\captured_test.jpg", jpg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    Open_camrea.release()

    
A_CAM_P()