import cv2
import numpy as np
import image_preprocessing


def find_contours(img):
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return img, contours, hierarchy


def draw_contours(img, contours):
    temp = np.ones(img.shape, dtype=np.uint8) * 255
    cv2.drawContours(temp, contours, -1, (0, 0, 0), 2)
    return temp


def select_parent_contours(hierarchy):
    delete_list = []
    c, row, col = hierarchy.shape
    for i in range(row):
        if hierarchy[0, i, 3] > 0:
            delete_list.append(i)
    return delete_list


def select_small_contours(contours):
    min_size = 400
    max_size = 20000
    delete_list = []
    for i in range(len(contours)):
        if (cv2.arcLength(contours[i], True) < min_size) or (cv2.arcLength(contours[i], True) > max_size):
            delete_list.append(i)
    return delete_list


def delete_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


def plate_location(contours):
    plate_list = []
    for i in range(len(contours)):
        # x, y, w, h = cv2.boundingRect(contours[i])
        # # print('i = ', i, '\n')
        # # print('w = ', w, '/ h = ', h, '\n')
        # if w > 2.5 * h:
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        rect = cv2.minAreaRect(contours[i])
        area = rect[1][0] * rect[1][1]
        if (area > 5000) and (area < 40000):
            if rect[1][1] > rect[1][0]:
                height = rect[1][0]
                width = rect[1][1]
            else:
                height = rect[1][1]
                width = rect[1][0]
            # print("area = ", area)
            # print("height = ", height)
            # print("width = ", width)
            # print("=======================")
            if (height * 2 < width) and (height * 5 > width):
                x, y, w, h = cv2.boundingRect(contours[i])
            # print('i = ', i, '\n')
            # print('w = ', w, '/ h = ', h, '\n')
                if w > h:
                    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
                    plate_list.append((x, y, x+w, y+h))
                    # print("x = ", x)
                    # print("y = ", y)
                    # print("========================")
            return plate_list


def plate_final_location(contours):
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if x != 0:
            return x, y, w, h


def canny_edge(gray):
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('binary', binary)
    edge = cv2.Canny(binary, 250, 300, 3)
    # cv2.imshow('edge', edge)
    # k = np.ones((5, 5), np.uint8)
    # r = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k)
    # 参数范围需要小心设置
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))  # 设定核的形状大小，由于车牌的字符是横向排列的，设定是横向膨胀
    car_dilation_2 = cv2.dilate(edge, kernel, iterations=6)  # 膨胀两次，保证字符区域全部连通起来，膨胀必须一步到位，否则再次腐蚀可能会将图像回复原装
    # cv2.imshow('car_dilation_2', car_dilation_2)
    car_erosion_4 = cv2.erode(car_dilation_2, kernel, iterations=6)  # 再腐蚀四次，尽可能多的去除小块碎片
    # cv2.imshow('car_erosion_41', car_erosion_4)
    car_dilation_4 = cv2.dilate(car_erosion_4, kernel, iterations=2)  # 再膨胀两次，保证膨胀总次数和腐蚀总次数相同
    # cv2.imshow('car_dilation_42', car_dilation_4)
    # 再进行Y方向的膨胀腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # 设定核的形状大小，纵向
    car_erosion_6 = cv2.erode(car_dilation_4, kernel, iterations=5)  # 再沿Y方向腐蚀两次，尽可能多的去除小块碎片
    # cv2.imshow('car_erosion_61', car_erosion_6)
    car_dilation_6 = cv2.dilate(car_erosion_6, kernel, iterations=5)  # 再膨胀两次，保证膨胀总次数和腐蚀总次数相同
    # cv2.imshow('car_dilation_62', car_dilation_6)  #经过六次腐蚀和六次膨胀，最终得到理想的结果
    return car_dilation_6


def edge_location(gray, init):
    edge = canny_edge(gray)
    img, contours, hierarchy = find_contours(edge)
    delete_small_list = select_small_contours(contours)
    contours = delete_contours(contours, delete_small_list)
    # temp = draw_contours(img, contours)
    edge_list = plate_location(contours)
    return edge_list


def hsv_location(binary):
    img, contours, hierarchy = find_contours(binary)
    # delete_parent_list = select_parent_contours(hierarchy)
    # contours = delete_contours(contours, delete_parent_list)
    delete_small_list = select_small_contours(contours)
    contours = delete_contours(contours, delete_small_list)
    hsv_list = plate_location(contours)
    return hsv_list


def rough_positioning(hsv_list, edge_list):
    plate_list = []
    min_distance = 100000
    j = 0
    p = 0
    q = 0
    if hsv_list is not None:
        if len(hsv_list) == 0:
            hsv_list = None
    if edge_list is not None:
        if len(edge_list) == 0:
            edge_list = None
    if hsv_list is None:
        if edge_list is None or len(edge_list) > 1:
            print('There is no plate!')
            return 0
        else:
            plate_list = edge_list
            return plate_list
    if edge_list is None:
        if hsv_list is None or len(hsv_list) > 1:
            print('There is no plate!')
            return 0
        else:
            plate_list = hsv_list
            return plate_list
    if (len(hsv_list) == 1) and (len(edge_list) == 1):
        if hsv_list[0][0] > edge_list[0][0]:
            x1 = edge_list[0][0]
        else:
            x1 = hsv_list[0][0]
        if hsv_list[0][1] > edge_list[0][1]:
            y1 = edge_list[0][1]
        else:
            y1 = hsv_list[0][1]
        if hsv_list[0][2] > edge_list[0][2]:
            x2 = hsv_list[0][2]
        else:
            x2 = edge_list[0][2]
        if hsv_list[0][3] > edge_list[0][3]:
            y2 = hsv_list[0][3]
        else:
            y2 = edge_list[0][3]
        plate_list.append((x1, y1, x2, y2))
    # elif (len(hsv_list) == 1) and (len(edge_list) == 0):
    #     plate_list = hsv_list
    # elif (len(hsv_list) == 0) and (len(edge_list) == 1):
    #     plate_list = edge_list
    elif (len(hsv_list) == 1) and (len(edge_list) > 1):
        focus_hsv_x, focus_hsv_y = get_focus(hsv_list[0][0], hsv_list[0][2], hsv_list[0][1], hsv_list[0][3])
        for i in edge_list:
            focus_x, focus_y = get_focus(i[0], i[2], i[1], i[3])
            distance = get_distance(focus_hsv_x, focus_hsv_y, focus_x, focus_y)
            if distance < min_distance:
                min_distance = distance
                j = i
        if hsv_list[0][0] > edge_list[j][0]:
            x1 = edge_list[j][0]
        else:
            x1 = hsv_list[0][0]
        if hsv_list[0][1] > edge_list[j][1]:
            y1 = edge_list[j][1]
        else:
            y1 = hsv_list[0][1]
        if hsv_list[0][2] > edge_list[j][2]:
            x2 = hsv_list[0][2]
        else:
            x2 = edge_list[j][2]
        if hsv_list[0][3] > edge_list[j][3]:
            y2 = hsv_list[0][3]
        else:
            y2 = edge_list[j][3]
        plate_list.append((x1, y1, x2, y2))
    elif (len(hsv_list) > 1) and (len(edge_list) == 1):
        focus_edge_x, focus_edge_y = get_focus(edge_list[0][0], edge_list[0][2], edge_list[0][1], edge_list[0][3])
        for i in hsv_list:
            focus_x, focus_y = get_focus(i[0], i[2], i[1], i[3])
            distance = get_distance(focus_edge_x, focus_edge_y, focus_x, focus_y)
            if distance < min_distance:
                min_distance = distance
                j = i
        if hsv_list[j][0] > edge_list[0][0]:
            x1 = edge_list[0][0]
        else:
            x1 = hsv_list[j][0]
        if hsv_list[j][1] > edge_list[0][1]:
            y1 = edge_list[0][1]
        else:
            y1 = hsv_list[j][1]
        if hsv_list[j][2] > edge_list[0][2]:
            x2 = hsv_list[j][2]
        else:
            x2 = edge_list[0][2]
        if hsv_list[j][3] > edge_list[0][3]:
            y2 = hsv_list[j][3]
        else:
            y2 = edge_list[0][3]
        plate_list.append((x1, y1, x2, y2))
    elif (len(hsv_list) > 1) and (len(edge_list) > 1):
        for m in hsv_list:
            focus_hsv_x, focus_hsv_y = get_focus(m[0], m[2], m[1], m[3])
            for n in edge_list:
                focus_edge_x, focus_edge_y = get_focus(n[0], n[2], n[1], n[3])
                distance = get_distance(focus_hsv_x, focus_hsv_y, focus_edge_x, focus_edge_y)
                if distance < min_distance:
                    min_distance = distance
                    p = m
                    q = n
        if hsv_list[m][0] > edge_list[n][0]:
            x1 = edge_list[n][0]
        else:
            x1 = hsv_list[m][0]
        if hsv_list[m][1] > edge_list[n][1]:
            y1 = edge_list[n][1]
        else:
            y1 = hsv_list[m][1]
        if hsv_list[m][2] > edge_list[n][2]:
            x2 = hsv_list[m][2]
        else:
            x2 = edge_list[n][2]
        if hsv_list[m][3] > edge_list[n][3]:
            y2 = hsv_list[m][3]
        else:
            y2 = edge_list[n][3]
        plate_list.append((x1, y1, x2, y2))
    return plate_list


def get_distance(x1, y1, x2, y2):
    distance = pow(pow(abs(x1 - x2), 2) + pow(abs(y1 - y2), 2), 0.5)
    return distance


def get_focus(x1, y1, x2, y2):
    focus_x = abs((x1 + x2) / 2)
    focus_y = abs((y1 + y2) / 2)
    return focus_x, focus_y


def rotate(img, angle, init_img, x, y):
    h = img.shape[0]
    w = img.shape[1]
    center = (w / 2 + x, h / 2 + y)
    h = init_img.shape[0]
    w = init_img.shape[1]
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate_img = cv2.warpAffine(init_img, rotate_matrix, (w, h))
    return rotate_img


def get_angle(img):
    sum_theta = 0
    img_canny = cv2.Canny(img, 50, 150, 3)
    lines = cv2.HoughLines(img_canny, 1, np.pi/180, 80)
    if lines is None:
        return 0
    for i in range(lines.shape[0]):
        theta = lines[i][0][1]
        sum_theta += theta
    average = sum_theta / lines.shape[0]
    angle = average / np.pi * 180 - 90
    return angle


def horizontal_correct(gray_img, hsv_img, init_img, x, y):
    ret, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angle = get_angle(binary)
    if angle == -1:
        print("No lines!")
        return 0
    print(angle)
    gray_plate = rotate(gray_img, angle, init_img, x, y)
    hsv_plate = rotate(hsv_img, angle, init_img, x, y)
    return gray_plate, hsv_plate


def plate_preprocessing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    r = cv2.erode(img, kernel, iterations=2)
    m = cv2.dilate(r, kernel, iterations=2)
    return m


def get_vertex(img):
    h = img.shape[0]
    w = img.shape[1]
    left_loop_num = 0
    right_loop_num = 0
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            m = w - i - 1
            n = h - j - 1
            if img[j, i] == 255 and left_loop_num == 0:
                left_loop_num = 1
                left = (j, i)
            if img[n, m] == 255 and right_loop_num == 0:
                right_loop_num = 1
                right = (n, m)
    top_list = get_top_point(img, h, w)
    bottom_list = get_bottom_point(img, h, w)
    m, n = get_max_distance(top_list, bottom_list, left, right)
    if left[0] < h / 2:
        left_top = left
        left_bottom = bottom_list[n]
        right_top = top_list[m]
        right_bottom = right
    else:
        left_top = top_list[m]
        left_bottom = left
        right_top = right
        right_bottom = bottom_list[n]
    return left_top, left_bottom, right_top, right_bottom


def get_top_point(img, h, w):
    top_list = []
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            if img[j, i] == 255:
                top_list.append((j, i))
                break
    return top_list


def get_bottom_point(img, h, w):
    bottom_list = []
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            m = h - j - 1
            if img[m, i] == 255:
                bottom_list.append((m, i))
                break
    return bottom_list


def get_max_distance(top_list, bottom_list, left, right):
    max_distance = 0
    threshold = abs(left[0] - right[0]) - 5
    for i in range(len(top_list)):
        for j in range(len(bottom_list)):
            top_to_left = get_distance(top_list[i][1], top_list[i][0], left[1], left[0])
            top_to_right = get_distance(top_list[i][1], top_list[i][0], right[1], right[0])
            bottom_to_left = get_distance(bottom_list[j][1], bottom_list[j][0], left[1], left[0])
            bottom_to_right = get_distance(bottom_list[j][1], bottom_list[j][0], right[1], right[0])
            if top_to_left > threshold and top_to_right > threshold and bottom_to_left > threshold and bottom_to_right > threshold:
                distance = get_distance(top_list[i][1], top_list[i][0], bottom_list[j][1], bottom_list[j][0])
                if distance > max_distance:
                    max_distance = distance
                    m = i
                    n = j
                    # m和n用作记录另外两个顶点的索引
    return m, n


def get_distance(x0, y0, x1, y1):
    x = abs(x0 - x1)
    y = abs(y0 - y1)
    distance = pow((x * x + y * y), 0.5)
    return distance


def tilt_correction(hsv_img, gray_img):
    left_top, left_bottom, right_top, right_bottom = get_vertex(hsv_img)
    pts1 = np.float32([[0, 0], [439, 0], [0, 139], [439, 139]])
    y0 = left_top[0]
    x0 = left_top[1]
    y1 = left_bottom[0]
    x1 = left_bottom[1]
    y2 = right_top[0]
    x2 = right_top[1]
    y3 = right_bottom[0]
    x3 = right_bottom[1]
    pts2 = np.float32([[x0, y0], [x2, y2], [x1, y1], [x3, y3]])
    m = cv2.getPerspectiveTransform(np.array(pts2), np.array(pts1))
    dst = cv2.warpPerspective(gray_img, m, (440, 140))
    return dst


def plate_locate(img):
    gray, binary, init = image_preprocessing.processed_image(img)
    hsv_list = hsv_location(binary)
    edge_list = edge_location(gray, init)
    plate_list = rough_positioning(hsv_list, edge_list)
    if plate_list == 0:
        return 0
    gray_img = gray[plate_list[0][1]:plate_list[0][3], plate_list[0][0]:plate_list[0][2]]
    hsv_img = binary[plate_list[0][1]:plate_list[0][3], plate_list[0][0]:plate_list[0][2]]
    init_img = init[plate_list[0][1]:plate_list[0][3], plate_list[0][0]:plate_list[0][2]]
    # hsv_img = plate_preprocessing(hsv_img)
    gray_plate, hsv_plate = horizontal_correct(gray_img, hsv_img, gray, plate_list[0][0], plate_list[0][1])
    gray_plate = gray_plate[plate_list[0][1]:plate_list[0][3], plate_list[0][0]:plate_list[0][2]]
    return gray_plate

