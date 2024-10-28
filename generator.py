# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy
import os

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

# 1.使用dlib自带的frontal_face_detector作为我们的人脸提取器
detector = dlib.get_frontal_face_detector()

# 2.使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class NoFaces(Exception):
    pass


list1 = os.listdir('presidents')
for i in range(0, len(list1)):
    image_path = os.path.join(list1[i])
    im = cv2.imread('presidents' + '/' + image_path)
    rects = detector(im, 1)
    filename = './presidents/' + image_path + '.txt'
    f = open(filename, 'w')
    for j in range(len(rects)):
        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[j]).parts()])
        im = im.copy()

        # 使用enumerate 函数遍历序列中的元素以及它们的下标
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            f.write(str(pos[0]))
            f.write(' ')
            f.write(str(pos[1]))
            f.write('\n')
    f.close()
