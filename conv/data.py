import cv2
import numpy as np
from os import listdir
import torch
from PIL import Image
import torchvision.transforms as transforms


# def read_data(image_path, txt):
#     f = open(txt, 'rb')
#     contents = f.readlines()
#     f.close()
#     x = []
#     y_ = []
#     for content in contents:
#         value = content.decode().split(" ")
#         img_path = image_path + '\\' + value[0]
#         img = Image.open(img_path)
#         img = np.array(img.convert('RGB'))
#         img = cv2.resize(img, (32, 32))
#         img = img / 255
#         x.append(img)
#         y_.append(value[1])
#     x = np.array(x)
#     y_ = np.array(y_)
#     y_ = y_.astype(np.int64)
#     return x, y_


def read_data(folder_path, label, x_train, y_train, x_test, y_test):
    character_folder = listdir(folder_path)
    n = 6
    for img in character_folder:
        img_path = folder_path + '\\' + img
        img = Image.open(img_path)
        img = np.array(img.convert('RGB'))
        img = cv2.resize(img, (32, 32))
        img = img / 255
        if n % 5 == 0:
            x_test.append(img)
            y_test.append(label)
        else:
            x_train.append(img)
            y_train.append(label)
        n = n + 1
    return x_train, y_train, x_test, y_test
