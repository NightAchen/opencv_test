import pandas as pd
from datetime import datetime

import cv2
import time
import numpy as np
import threading
import queue
from collections import deque
from ultralytics import YOLO

frame_queue = deque(maxlen=2)  # 初始化为双端队列
# 在主程序中初始化一个列表来存储识别事件
recognized_faces = []

model = YOLO("weights/best.pt").to('cuda')
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('D:/Py_prj/opencv_Face/mycodetest/opencv/trainer/trainer.yml')


def get_name_by_label(label):
    # 本地文件 'names.txt'，每行包含一个标签和名字，用逗号分隔
    names_file = 'names.txt'
    names = {}

    try:
        with open(names_file, 'r') as file:
            for line in file:
                label_from_file, name = line.strip().split(',')
                names[int(label_from_file)] = name
    except FileNotFoundError:
        print(f"文件 {names_file} 未找到。")
        return 'Unknown'
    except Exception as e:
        print(f"发生错误: {e}")
        return 'Unknown'

    return names.get(label, 'Unknown')


# YOLO 人脸检测模块
def yolo_face_detect(frame_queue, stop_event, recognized_faces):
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    # # 再设置高清采集分辨率
    # cap.set(3, 1920)  # 设置水平分辨率
    # cap.set(4, 1080)  # 设置垂直分辨率
    # cap.set(cv2.CAP_PROP_FPS, 30)  # 限制帧率为15FPS

    start_time = time.time()
    count = 0
    while not stop_event.is_set() and cap.isOpened():
        count = count + 1
        success, frame = cap.read()
        if success:
            results = model(frame, conf=0.45, classes=0, device='cuda')
            annotated_frame = results[0].plot()
            # 遍历检测结果中的每个人脸
            for result in results:
                # 获取边界框对象
                boxes = result.boxes
                # 遍历每个边界框
                for box in boxes:
                    # 获取边界框的坐标
                    x1, y1, x2, y2 = box.xyxy[0]
                    # 将坐标转换为整数
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    face = frame[y1:y2, x1:x2]
                    frame_queue.append(face)  # 使用 append 方法添加到双端队列

                    if frame_queue:
                        try:
                            face_frame = frame_queue.popleft()  # 使用 popleft 方法从双端队列中取出
                            gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                            label, confidence = recogizer.predict(gray_face)
                            print(f"Recognized label: {label} with confidence: {confidence}")
                            # 在YOLO人脸检测模块中调用函数获取标签对应的名字
                            if confidence < 80:
                                name = get_name_by_label(label)
                            else:
                                name = 'Unknown'
                            # 记录识别的时间和名字
                            recognized_faces.append({
                                'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Name': name
                            })
                            # 将处理后的图像放回原图像
                            frame[y1:y2, x1:x2] = annotated_frame[y1:y2, x1:x2]
                            # 在帧上绘制名称
                            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)

                            cv2.imshow('OpenCV Face Recognition', frame)
                            if time.time() - start_time >= 1:
                                start_time = time.time()
                                print("display_frame_1秒内帧数", count)
                                count = 0

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                stop_event.set()
                                break
                        except queue.Empty:
                            continue
    cap.release()
    cv2.destroyAllWindows()


# 在程序结束前保存识别事件到Excel表格
def save_recognitions_to_excel(recognized_faces):
    try:
        df = pd.DataFrame(recognized_faces)
        df.to_excel('recognized_faces.xlsx', index=False)
        print("Excel file saved successfully.")  # 打印确认信息
    except Exception as e:
        print(f"Error saving Excel file: {e}")  # 打印错误信息

# 主程序
if __name__ == '__main__':
    stop_event = threading.Event()

    # 创建并启动YOLO人脸检测线程
    yolo_thread = threading.Thread(target=yolo_face_detect, args=(frame_queue, stop_event, recognized_faces))
    yolo_thread.start()

    # 等待线程结束
    yolo_thread.join()

    save_recognitions_to_excel(recognized_faces)