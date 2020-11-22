<<<<<<< HEAD
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import config

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


class myFilter(object):
    def __init__(self) -> None:
        pass

    def __call__(self, img):
        # PIL转为OpenCV格式
        img_cv = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

        # 限制对比度的自适应直方图均衡化
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
        channels = cv.split(img_cv)
        for i in range(3):
            channels[i] = clahe.apply(channels[i])
        cv.merge(channels, img_cv)

        # OpenCV格式转回PIL
        img = Image.fromarray(cv.cvtColor(img_cv, cv.COLOR_RGB2BGR))

        # img.show()

        return img


def getTransform():
    return transforms.Compose(
        [
            myFilter(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.Resize(50),
            transforms.CenterCrop(50),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
                )
        ]
    )


def getTransformForTestAndPredict():
    return transforms.Compose(
        [
            myFilter(),
            transforms.Resize(50),
            transforms.CenterCrop(50),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
                )
        ]
    )


def getData(batch_size=8, num_workers=2):
    transform = getTransform()
    transform_for_test = getTransformForTestAndPredict()
    train_set = ImageFolder(
        root=config.DATA_TRAIN, transform=transform
    )
    test_set = ImageFolder(
        root=config.DATA_TEST, transform=transform_for_test
    )
    # print(os.listdir(config.DATA_TRAIN))
    # print(len(train_set))
    # print(train_set.class_to_idx)
    # for i in train_set:
    #     print(i[1])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # print(train_loader)
    return train_loader, test_loader


if __name__ == "__main__":
    # img_dark = Image.open("img_dark.jpg")
    # img_light = Image.open("img_light.jpg")
    transform = getTransform()
    # img_dark = trans(img_dark)
    # img_light = trans(img_light)
    # img_dark.save("img_dark_res1.jpg")
    # img_light.save("img_light_res1.jpg")
    ls = os.listdir(os.path.join(config.DATA_TRAIN, "me"))
    for i in range(10):
        img = Image.open(os.path.join(config.DATA_TRAIN, "me", ls[i]))
        img = transform(img)
        img.save("train{}.jpg".format(i))

    transform = getTransformForTestAndPredict()
    ls = os.listdir(os.path.join(config.DATA_TEST, "me"))
    for i in range(10):
        img = Image.open(os.path.join(config.DATA_TEST, "me", ls[i]))
        img = transform(img)
        img.save("test{}.jpg".format(i))
=======
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import config

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


class myFilter(object):
    def __init__(self) -> None:
        pass

    def __call__(self, img):
        # PIL转为OpenCV格式
        img_cv = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

        # 限制对比度的自适应直方图均衡化
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
        channels = cv.split(img_cv)
        for i in range(3):
            channels[i] = clahe.apply(channels[i])
        cv.merge(channels, img_cv)

        # OpenCV格式转回PIL
        img = Image.fromarray(cv.cvtColor(img_cv, cv.COLOR_RGB2BGR))

        # img.show()

        return img


def getTransform():
    return transforms.Compose(
        [
            myFilter(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.Resize(50),
            transforms.CenterCrop(50),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
                )
        ]
    )


def getTransformForTestAndPredict():
    return transforms.Compose(
        [
            myFilter(),
            transforms.Resize(50),
            transforms.CenterCrop(50),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
                )
        ]
    )


def getData(batch_size=8, num_workers=2):
    transform = getTransform()
    transform_for_test = getTransformForTestAndPredict()
    train_set = ImageFolder(
        root=config.DATA_TRAIN, transform=transform
    )
    test_set = ImageFolder(
        root=config.DATA_TEST, transform=transform_for_test
    )
    # print(os.listdir(config.DATA_TRAIN))
    # print(len(train_set))
    # print(train_set.class_to_idx)
    # for i in train_set:
    #     print(i[1])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # print(train_loader)
    return train_loader, test_loader


if __name__ == "__main__":
    # img_dark = Image.open("img_dark.jpg")
    # img_light = Image.open("img_light.jpg")
    transform = getTransform()
    # img_dark = trans(img_dark)
    # img_light = trans(img_light)
    # img_dark.save("img_dark_res1.jpg")
    # img_light.save("img_light_res1.jpg")
    ls = os.listdir(os.path.join(config.DATA_TRAIN, "me"))
    for i in range(10):
        img = Image.open(os.path.join(config.DATA_TRAIN, "me", ls[i]))
        img = transform(img)
        img.save("train{}.jpg".format(i))

    transform = getTransformForTestAndPredict()
    ls = os.listdir(os.path.join(config.DATA_TEST, "me"))
    for i in range(10):
        img = Image.open(os.path.join(config.DATA_TEST, "me", ls[i]))
        img = transform(img)
        img.save("test{}.jpg".format(i))
>>>>>>> 2f1e176... first commit
