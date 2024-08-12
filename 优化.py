from datetime import datetime
import cv2
import threading
from ultralytics import YOLO
import pandas as pd
import queue

rtsp_url = "rtsp://admin:Aa123456@192.168.10.245:1554/Streaming/Channels/101"

class_names = {
    1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


model = YOLO("weights/best.pt").to('cuda')
model_person = YOLO("weights/yolov8n.pt").to('cuda')
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('D:/Py_prj/opencv_Face/mycodetest/opencv/trainer/trainer.yml')

# 全局变量
frame_lock = threading.Lock()
recognized_faces = []  # 在主程序中初始化一个列表来存储识别事件
detected_classes = []  # 包含检测结果的列表
frame_queue = queue.Queue(maxsize=1000)  # 设置队列的最大大小
frame_buffer = [None, None]
current_buffer = 0

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

# 视频捕获线程
def capture_video():
    print("capture_video thread ID:", threading.get_ident())
    global frame_buffer, current_buffer
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS, 60)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    # # 再设置高清采集分辨率
    # cap.set(3, 1280)  # 设置水平分辨率
    # cap.set(4, 720)  # 设置垂直分辨率
    while not stop_event.is_set():
        success, new_frame = cap.read()
        if not success:
            break
        with frame_lock:
            frame_buffer[current_buffer] = new_frame
            current_buffer = (current_buffer + 1) % 2
    cap.release()

# 图像显示线程
def display_frame():
    print("display_frame thread ID:", threading.get_ident())
    global frame_buffer, current_buffer
    count = 0
    while not stop_event.is_set():
        with frame_lock:
            frame = frame_buffer[(current_buffer + 1) % 2]
        if frame is not None:
            results = model(frame, conf=0.45, classes=0, device='cuda', verbose=False)
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
                    face_frame = frame[y1:y2, x1:x2]
                    gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                    label, confidence = recogizer.predict(gray_face)
                    # 在YOLO人脸检测模块中调用函数获取标签对应的名字
                    if confidence < 80:
                        name = get_name_by_label(label)
                    else:
                        name = 'unknow'
                    count += 1  # 递增计数器
                    if count == 6:  # 检查计数器是否等于7
                        recognized_faces.append({
                            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Name': name
                        })
                        save_recognitions_to_excel('face.xlsx', recognized_faces)
                        count = 0
                    # 将处理后的图像放回原图像
                    frame[y1:y2, x1:x2] = annotated_frame[y1:y2, x1:x2]
                    # 在帧上绘制名称
                    cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                    frame_queue.put_nowait(frame)

# 物品检测
def item_detection(stop_event):
    count = 0
    annotated_frame2 = None
    while not stop_event.is_set():
        try:
            people_frame = frame_queue.get(timeout=1)  # 阻塞直到有元素可用或超时
            results = model_person(people_frame, conf=0.45, device='cuda', verbose=False)
            for frame_results in results:
                # 获取每一帧的boxes属性
                boxes = frame_results.boxes
                for box in boxes:
                    # 获取类别索引，并将其转换为类别名称
                    class_index = int(box.cls)
                    class_name = class_names.get(class_index)  # 如果索引不存在于映射中，则返回' unknown'
                    if class_name:
                        count += 1  # 递增计数器
                        if count == 6:  # 检查计数器是否等于7
                            detected_classes.append({
                                'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'belongings': class_name
                            })
                            save_recognitions_to_excel('belongings.xlsx', detected_classes)
                            count = 0  # 重置计数器
            annotated_frame = results[0].plot()
            # # 遍历检测结果中的每个人
            # for result in results:
            #     # 获取边界框对象
            #     boxes = result.boxes
            #     # 遍历每个边界框
            #     for box in boxes:
            #         # 获取边界框的坐标
            #         x1, y1, x2, y2 = box.xyxy[0]
            #         cls = box.cls[0]
            #         # 将坐标转换为整数
            #         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            #         roi = people_frame[y1:y2, x1:x2]
            #         results2 = model_person(roi, conf=0.6, classes=67, device='cuda', verbose=False)
            #         annotated_frame2 = results2[0].plot()
            #         recognized_faces.append({
            #             'belongings': cls
            #         })
            #         # 将处理后的图像放回原图像
            #         annotated_frame[y1:y2, x1:x2] = annotated_frame2
            cv2.imshow('OpenCV Face Recognition', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except queue.Empty:
            continue
    cv2.destroyAllWindows()

# 在程序结束前保存识别事件到Excel表格
def save_recognitions_to_excel(excel_name, recognized):
    # try:
    df = pd.DataFrame(recognized)
    df.to_excel(excel_name, index=False)
    #     print(f"{excel_name} file saved successfully.")  # 打印确认信息
    # except Exception as e:
    #     print(f"Error saving {excel_name} file: {e}")  # 打印错误信息


if __name__ == '__main__':
    print("Main thread ID:", threading.get_ident())

    stop_event = threading.Event()
    # 启动视频捕获线程
    capture_thread = threading.Thread(target=capture_video)
    capture_thread.start()

    # opencv + yolo 人脸识别线程
    display_thread = threading.Thread(target=display_frame)
    display_thread.start()

    # yolo物品检测线程
    item_detection_thread = threading.Thread(target=item_detection, args=(stop_event, ))
    item_detection_thread.start()

    # 等待线程结束
    capture_thread.join()
    display_thread.join()
    item_detection_thread.join()

