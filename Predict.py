<<<<<<< HEAD
import os

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from FaceRecNet import FaceRecNet
from Preprocess import getTransformForTestAndPredict

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
FACE_LABEL = os.listdir("data\\train")


def predictModel(image):
    transform = getTransformForTestAndPredict()
    image = transform(image)
    image = image.view(-1, 3, 50, 50)
    net = FaceRecNet().to(DEVICE)
    net.load_state_dict(
        torch.load(os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL))
    )
    output = net(image.to(DEVICE))
    print("-----output-----")
    print(output)
    print(np.var(output[0, :].cpu().detach().numpy()))
    print(output.max(1, keepdim=True))
    print(output.max(1, keepdim=True)[1])
    var = np.var(output[0, :].cpu().detach().numpy())
    if var <= 3:
        print("----pred.item()----")
        print(-1)
        print('\n')
        return -1
    pred = output.max(1, keepdim=True)[1]
    print("----pred.item()----")
    print(pred.item())
    print("\n")
    return pred.item()


def paintName(im, name, pos, color):
    img_PIL = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    # 引用字体库
    font = ImageFont.truetype('msyh.ttc', 20)
    if name == "陌生人，unknown":
        fillColor = (255, 0, 0)
    else:
        fillColor = color
    position = pos
    draw = ImageDraw.Draw(img_PIL)
    # 写上人脸对应的人名
    draw.text(position, name, font=font, fill=fillColor)
    img = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)
    return img


def cvToPIL(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def catchFace(frame):
    cascade = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_boxes = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64)
        )
    if len(face_boxes) > 0:
        for face_box in face_boxes:
            x, y, w, h = face_box
            image = frame[y: y + h, x: x + w]
            # openCV 转 PIL格式图片
            PIL_image = cvToPIL(image)
            # 使用模型进行人脸识别
            label = predictModel(PIL_image)
            if label == -1:
                name = "陌生人，unknown"
                color = (0, 0, 255)
            else:
                name = FACE_LABEL[label]
            cv.rectangle(
                frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            # 将人脸对应人名写到图片上, 可能是中文名所以需要加载中文字体库
            color = (0, 255, 0)
            frame = paintName(
                frame, name, (x-10, y+h+10), color
                )
    return frame


def recognizeFromVideo(window_name='FaceRecognize', camera_idx=0):
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(camera_idx)
    while cap.isOpened():
        ready, frame = cap.read()
        if not ready:
            break
        catch_frame = catchFace(frame)
        cv.imshow(window_name, catch_frame)
        c = cv.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    recognizeFromVideo()
=======
import os

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from FaceRecNet import FaceRecNet
from Preprocess import getTransformForTestAndPredict

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
FACE_LABEL = os.listdir("data\\train")


def predictModel(image):
    transform = getTransformForTestAndPredict()
    image = transform(image)
    image = image.view(-1, 3, 50, 50)
    net = FaceRecNet().to(DEVICE)
    net.load_state_dict(
        torch.load(os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL))
    )
    output = net(image.to(DEVICE))
    print("-----output-----")
    print(output)
    print(np.var(output[0, :].cpu().detach().numpy()))
    print(output.max(1, keepdim=True))
    print(output.max(1, keepdim=True)[1])
    var = np.var(output[0, :].cpu().detach().numpy())
    if var <= 3:
        print("----pred.item()----")
        print(-1)
        print('\n')
        return -1
    pred = output.max(1, keepdim=True)[1]
    print("----pred.item()----")
    print(pred.item())
    print("\n")
    return pred.item()


def paintName(im, name, pos, color):
    img_PIL = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    # 引用字体库
    font = ImageFont.truetype('msyh.ttc', 20)
    if name == "陌生人，unknown":
        fillColor = (255, 0, 0)
    else:
        fillColor = color
    position = pos
    draw = ImageDraw.Draw(img_PIL)
    # 写上人脸对应的人名
    draw.text(position, name, font=font, fill=fillColor)
    img = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)
    return img


def cvToPIL(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def catchFace(frame):
    cascade = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_boxes = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64)
        )
    if len(face_boxes) > 0:
        for face_box in face_boxes:
            x, y, w, h = face_box
            image = frame[y: y + h, x: x + w]
            # openCV 转 PIL格式图片
            PIL_image = cvToPIL(image)
            # 使用模型进行人脸识别
            label = predictModel(PIL_image)
            if label == -1:
                name = "陌生人，unknown"
                color = (0, 0, 255)
            else:
                name = FACE_LABEL[label]
            cv.rectangle(
                frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            # 将人脸对应人名写到图片上, 可能是中文名所以需要加载中文字体库
            color = (0, 255, 0)
            frame = paintName(
                frame, name, (x-10, y+h+10), color
                )
    return frame


def recognizeFromVideo(window_name='FaceRecognize', camera_idx=0):
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(camera_idx)
    while cap.isOpened():
        ready, frame = cap.read()
        if not ready:
            break
        catch_frame = catchFace(frame)
        cv.imshow(window_name, catch_frame)
        c = cv.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    recognizeFromVideo()
>>>>>>> 2f1e176... first commit
