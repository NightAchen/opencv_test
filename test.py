import os
import cv2
import time
import threading
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency

# 检查文件是否存在
file_path = 'weights/best.pt'
if os.path.exists(file_path):
    print(f"文件存在于 {file_path}")
else:
    print(f"文件不存在于 {file_path}，请检查路径是否正确")


# 加载 YOLOv8 模型
model = YOLO("weights/opencvface/best.pt")  #预测人脸的大模型
model.to('cuda')
model_person = YOLO("weights/yolov8n.pt")  #预测人类的大模型
model_person.to('cuda')

# 加载训练好的人脸识别模型
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read("D:/Py_prj/opencv_Face/mycodetest/opencv/trainer/trainer.yml")
names = []

# RTSP流地址
#rtsp_url = "rtsp://admin:Gcc@123456.@192.168.1.64:554/Streaming/Channels/101"
rtsp_url = "rtsp://admin:Aa123456@175.0.136.23:1554/Streaming/Channels/101"

# 获取摄像头内容，参数 0 表示使用默认的摄像头
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
# 再设置高清采集分辨率
cap.set(3,1920)  # 设置水平分辨率
cap.set(4,1080)  # 设置垂直分辨率

start_time = time.time()
count = 0
while cap.isOpened():
    count = count + 1
    loop_start = getTickCount()
    success, frame = cap.read()  # 读取摄像头的一帧图像

    if success:
        # 定义ROI的坐标，例如：x, y, width, height
        x, y, w, h = (1920 - 1000) // 2, (1080 - 900) // 2, 1000, 900  # 坐标（x,y）
        # 获取ROI
        roi = frame[y:y + h, x:x + w]

        # 对图像进行预测，并指定置信度阈值,只能识别人,手机，笔记本电脑  classes=[0,67,68],
        results = model(roi, conf=0.45,  device='cuda')
        annotated_frame = results[0].plot()
        results2 = model_person(annotated_frame, conf=0.6, classes=0, device='cuda')
        annotated_frame2 = results2[0].plot()

        # 将处理后的ROI放回原图像
        frame[y:y + h, x:x + w] = annotated_frame2

        # 在ROI周围绘制绿色框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 全屏显示
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('img', frame)
        if time.time() - start_time >= 1:
            start_time = time.time()
            print("display_frame_1秒内帧数", count)
            count = 0

    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口