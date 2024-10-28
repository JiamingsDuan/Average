# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy
import os

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PICTURE_PATH = './presidents'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

list1 = os.listdir(PICTURE_PATH)
for i in range(0, len(list1)):
    image_path = os.path.join(list1[i])
    print(image_path)
    im = cv2.imread(PICTURE_PATH + '/' + image_path)
    rects = detector(im, 1)
    if len(rects) >= 1:
        print("{} faces detected".format(len(rects)))
    if len(rects) == 0:
        pass
    name = os.path.splitext(image_path)[0]
    filename = './presidents/' + name + '.doc'
    f = open(filename, 'w')
    for j in range(len(rects)):
        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[j]).parts()])
        im = im.copy()

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            f.write(str(pos[0]))
            f.write(' ')
            f.write(str(pos[1]))
            f.write('\n')
            cv2.circle(im, pos, 3, color=(255, 255, 255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, str(idx + 1), pos, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    f.close()
    cv2.imwrite('./presidents/' + name + '.jpg', im)
