import cv2
import numpy as np


def load_image(image_name, color):
    img = cv2.imread(image_name, color)
    img1 = img_filter(img)
    return img1


def img_filter(img):
    median = cv2.medianBlur(img, 3)
    gauss = cv2.GaussianBlur(median, (3, 3), 0, 0)
    return gauss


def img_binary(image_name):
    img_hsv = load_image(image_name, 1)
    hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 90, 90])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    k = np.ones((3, 3), np.uint8)
    r = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return r


def processed_image(img):
    gray = load_image(img, 0)
    binary = img_binary(img)
    init = load_image(img, 1)
    return gray, binary, init
