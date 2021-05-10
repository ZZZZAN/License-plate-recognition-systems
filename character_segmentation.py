import cv2
import numpy as np
import license_plate_locate
from os import listdir
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from PIL import Image
import tensorflow as tf


def horizontal_projection(h, w, img):
    pro_list = []
    for i in range(0, h - 1):
        pixel = 0
        for j in range(0, w - 1):
            if img[i, j] == 255:
                pixel = pixel + 1
        pro_list.append(pixel)
    dst = np.zeros((h, w, 1), np.uint8)
    dst.fill(255)
    for i in range(0, h - 1):
        for j in range(0, pro_list[i] - 1):
            dst[i, j] = 0
    trough1, trough2 = judge_trough(h, w, pro_list)
    return trough1, trough2


def judge_trough(h, w, pro_list):
    trough1 = 0
    trough2 = 0
    for i in range(1, h - 1):
        if pro_list[i] < w / 5:
            if (i - trough1) > h / 3:
                trough2 = i
                break
            else:
                trough1 = i
    return trough1, trough2


def vertical_projection(img):
    h = img.shape[0]
    w = img.shape[1]
    pro_list = []
    for i in range(0, w - 1):
        pixel = 0
        for j in range(0, h - 1):
            if img[j, i] == 255:
                pixel = pixel + 1
        pro_list.append(pixel)
    dst = np.zeros((h, w, 1), np.uint8)
    dst.fill(255)
    for i in range(0, w - 1):
        for j in range(0, pro_list[i] - 1):
            dst[j, i] = 0
    cv2.imshow('dst', dst)
    peak_list = judge_peak(w, pro_list)
    return peak_list


def judge_peak(w, pro_list):
    peak1 = 1
    peak2 = 1
    peak_list = []
    if pro_list[0] != 0:
        peak1 = 0
    for i in range(0, w - 1):
        if pro_list[i] == 0:
            if (i - peak1) > 1:
                peak2 = i
                r = [peak1, peak2]
                peak_list.append(r)
                peak1 = i
                peak2 = i
            peak1 = i
    if len(peak_list) == 0:
        return 0
    peak_list = combine_character(peak_list, pro_list)
    if peak_list is None:
        peak_list == 0
        return 0
    if peak_list == 0:
        return 0
    peak_list = delete_peak(peak_list, pro_list)
    return peak_list


def combine_character(peak_list, pro_list):
    m = 0
    for i in range(0, len(peak_list)):
        if peak_list[i][1] - peak_list[i][0] < 10:
            peak = int(peak_list[i][0] + (peak_list[i][1] - peak_list[i][0]) / 2)
            if pro_list[peak] < 20:
                m = i - 1
    if m == 0:
        return 0
    peak = peak_list[m][1] - peak_list[m][0]
    temp_peak_list = peak_list[:]
    for j in range(0, m - 1):
        n = m - 1 - j
        peak1 = temp_peak_list[n][0]
        if n == (m - 1):
            peak2 = temp_peak_list[n][1]
        peak_chinese = peak2 - peak1
        if abs(peak - peak_chinese) < 10:
            r = [peak1, peak2]
            peak_list.append(r)
            peak_list.remove([temp_peak_list[n][0], temp_peak_list[n][1]])
            return peak_list
        else:
            peak_list.remove([temp_peak_list[n][0], temp_peak_list[n][1]])
    return peak_list


def delete_peak(peak_list, pro_list):
    temp_peak_list = peak_list[:]
    for i in range(0, len(temp_peak_list)):
        if temp_peak_list[i][1] - temp_peak_list[i][0] < 10:
            peak = int(temp_peak_list[i][0] + (temp_peak_list[i][1] - temp_peak_list[i][0]) / 2)
            if pro_list[peak] < 20 or (temp_peak_list[i][1] - temp_peak_list[i][0]) < 5:
                peak_list.remove([temp_peak_list[i][0], temp_peak_list[i][1]])
    return peak_list


def character_segmentation(peak_list, img):
    h = img.shape[0]
    x = []
    for i in range(0, len(peak_list)):
        temp = img[0:h, peak_list[i][0]:peak_list[i][1]]
        temp_name = r'D:\Graduation_project\predict_img' + '\\' + str(i) + '.png'
        cv2.imwrite(temp_name, temp)


model_save_path = r'D:\Graduation_project\conv\checkpoint\LeNet5.ckpt'

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=10, kernel_size=(3, 3),
                         activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.c2 = Conv2D(filters=20, kernel_size=(3, 3),
                         activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.flatten = Flatten()
        self.f1 = Dense(1000, activation='sigmoid')
        self.f2 = Dense(1000, activation='sigmoid')
        self.f3 = Dense(65, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y

model = LeNet5()
model.load_weights(model_save_path)


def main():
    img_name = r'test3.jpg'
    gray_plate = license_plate_locate.plate_locate(img_name)
    if isinstance(gray_plate, int):
        return 0
    plate_height = gray_plate.shape[0]
    plate_width = gray_plate.shape[1]
    ret, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    trough1, trough2 = horizontal_projection(plate_height, plate_width, binary_plate)
    plate = binary_plate[trough1:trough2, 0:plate_width]
    peak_list = vertical_projection(plate)
    if peak_list == 0:
        return 0
    character_segmentation(peak_list, plate)
    cv2.waitKey(0)
    # character_folder = listdir(r'D:\Graduation_project\predict_img')
    # labels = []
    # for i in character_folder:
    #     character_path = r'D:\Graduation_project\predict_img' + '\\' + str(i)
    #     img = Image.open(character_path)
    #     img = img.resize((32, 32), Image.ANTIALIAS)
    #     img_arr = np.array(img.convert('RGB'))
    #     img_arr = img_arr / 255
    #     x_predict = img_arr[tf.newaxis, ...]
    #     result = model.predict(x_predict)
    #     pred = tf.argmax(result, axis=1)
    #     res = pred[0]
    #     result = res.numpy()
    #     labels.append(result)
    # match = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
    #          13: '川', 14: 'D', 15: 'E', 16: '鄂',
    #          17: 'F', 18: 'G', 19: '赣', 20: '甘', 21: '贵', 22: '桂', 23: 'H', 24: '黑', 25: '沪', 26: 'J', 27: '冀', 28: '津',
    #          29: '京', 30: '吉', 31: 'K', 32: 'L', 33: '辽',
    #          34: '鲁', 35: 'M', 36: '蒙', 37: '闽', 38: 'N', 39: '宁', 40: 'P', 41: 'Q', 42: '青', 43: '琼', 44: 'R', 45: 'S',
    #          46: '陕', 47: '苏', 48: '晋', 49: 'T', 50: 'U',
    #          51: 'V', 52: 'W ', 53: '皖', 54: 'X', 55: '湘', 56: '新', 57: 'Y', 58: '豫', 59: '渝', 60: '粤', 61: '云',
    #          62: 'Z',
    #          63: '藏', 64: '浙'
    #          }
    # print('\n车牌识别结果为: ', end='')
    # for j in range(len(labels)):
    #     print(match[labels[j]], end='')


if __name__ == '__main__':
    main()