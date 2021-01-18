import os
import time

import cv2 as cv

import config


def loggingFromVideo(tag, windows_name='CatchFace', camera_id=0):
    cv.namedWindow(windows_name)

    cap = cv.VideoCapture(camera_id)
    while cap.isOpened():
        # 读取一帧
        ok, frame = cap.read()
        if not ok:
            break
        # 抓取人脸
        catchFace(frame, tag)
        cv.imshow(windows_name, frame)
        # 按'q' 退出
        if cv.waitKey(1) == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv.destroyAllWindows()


def catchFace(frame, tag):
    # 向OpenCV传入要使用的分类器
    cascade = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # 识别出人脸后画出的框的颜色
    color = (0, 255, 0)
    # 将图像转换为灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rects = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=10, minSize=(64, 64))
    num = 1
    if len(face_rects) > 0:
        # 图片帧中有多张图片，框出每个人脸
        for face_rect in face_rects:
            x, y, w, h = face_rect
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            # 保存人脸图像
            saveFace(image, tag, num)
            cv.rectangle(
                frame, (x - 10, y - 10), (x + w + 10, y + w + 10), color, 2)
            num += 1


def saveFace(image, tag, num):
    # 将人脸图片存放到指定目录
    if not os.path.exists(config.DATA_TRAIN):
        os.mkdir(config.DATA_TRAIN)
    if not os.path.exists(os.path.join(config.DATA_TRAIN, tag)):
        print(os.path.join(config.DATA_TRAIN, tag))
        os.mkdir(os.path.join(config.DATA_TRAIN, tag))
    img_name = os.path.join(config.DATA_TRAIN, tag,
                            "{}_{}.jpg".format(int(time.time()), num)
                            )
    # print(img_name)
    cv.imwrite(img_name, image)


if __name__ == "__main__":
    loggingFromVideo("xxx")
