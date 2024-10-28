# ---* Feature calculation *--
import math
import pandas as pd
from pandas import DataFrame


def generate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def generate_specific(numerator, denominator):
    return format(numerator / denominator, '.4f')


def generate_median(x1, y1, x2, y2):
    return (x1 + x2)*0.5, (y1 + y2)*0.5


def average(n, m):
    return (n + m)*0.5


def generate_features(feature_list):
    # 【1】face
    landmark_x = feature_list[1:136:2]
    landmark_y = feature_list[2:137:2]
    # 眉间心
    eyebrow_mid_x, eyebrow_mid_y = generate_median(landmark_x[19], landmark_y[19], landmark_x[24], landmark_y[24])
    length_of_face = generate_distance(eyebrow_mid_x, eyebrow_mid_y, landmark_x[8], landmark_y[8])
    width_of_face = generate_distance(landmark_x[0], landmark_y[0], landmark_x[16], landmark_y[16])
    RF = generate_specific(width_of_face, length_of_face)  # 宽高比
    cheekbones = generate_distance(landmark_x[1], landmark_y[1], landmark_x[15], landmark_y[15])
    eyelid = generate_distance(landmark_x[27], landmark_y[27], landmark_x[62], landmark_y[62])
    WHR = generate_specific(cheekbones, eyelid)  # 面部宽高比 width / height
    # 【2】eye
    width_l_eye = generate_distance(landmark_x[36], landmark_y[36], landmark_x[39], landmark_y[39])
    width_R_eye = generate_distance(landmark_x[42], landmark_y[42], landmark_x[45], landmark_y[45])
    eye_width_ave = average(width_l_eye, width_R_eye)  # 眼宽均值
    height_l_eye = generate_distance(landmark_x[38], landmark_y[38], landmark_x[40], landmark_y[40])
    height_R_eye = generate_distance(landmark_x[43], landmark_y[43], landmark_x[47], landmark_y[47])
    eye_height_ave = average(height_l_eye, height_R_eye)  # 眼高均值
    L_pupil_x, L_pupil_y = generate_median(landmark_x[38], landmark_y[38], landmark_x[41], landmark_y[41])  # 左瞳孔
    R_pupil_x, R_pupil_y = generate_median(landmark_x[43], landmark_y[43], landmark_x[46], landmark_y[46])  # 右瞳孔
    dist_of_pupil = generate_distance(L_pupil_x, L_pupil_y, R_pupil_x, R_pupil_y)  # 瞳间距
    RHL = generate_specific(eye_width_ave, eye_height_ave)  # 眼睛宽高均值比
    REF = generate_specific(dist_of_pupil, width_of_face)  # 瞳间距占脸宽
    REL = generate_specific(eye_width_ave, width_of_face)  # 眼睛宽比脸宽
    REH = generate_specific(eye_height_ave, length_of_face)  # 眼睛高比脸长
    # 【3】nose
    height_of_nose = generate_distance(landmark_x[27], landmark_y[27], landmark_x[33], landmark_y[33])
    width_of_nose = generate_distance(landmark_x[31], landmark_y[31], landmark_x[35], landmark_y[35])
    RNO = generate_specific(width_of_nose, height_of_nose)  # 鼻子宽高比
    RNL = generate_specific(height_of_nose, length_of_face)  # 鼻子高占脸长
    RHO = generate_specific(width_of_nose, width_of_face)  # 鼻子宽占脸宽
    # 【4】jaw
    L_mouse_line_lip = generate_distance(landmark_x[3], landmark_y[3], landmark_x[51], landmark_y[51])
    R_mouse_line_lip = generate_distance(landmark_x[13], landmark_y[13], landmark_x[51], landmark_y[51])
    mouse_line_lip = average(L_mouse_line_lip, R_mouse_line_lip)  # 上唇中心到脸边缘
    tip_of_chin = generate_distance(landmark_x[8], landmark_y[8], landmark_x[66], landmark_y[66])  # TOC
    jaw_dist_01 = generate_distance(landmark_x[12], landmark_y[12], landmark_x[4], landmark_y[4])  # JAW1
    jaw_dist_02 = generate_distance(landmark_x[11], landmark_y[11], landmark_x[5], landmark_y[5])  # JAW2
    jaw_dist_03 = generate_distance(landmark_x[10], landmark_y[10], landmark_x[6], landmark_y[6])  # JAW3
    jaw_dist_04 = generate_distance(landmark_x[9], landmark_y[9], landmark_x[7], landmark_y[7])  # JAW4
    RML = generate_specific(mouse_line_lip, width_of_face)  # 上唇中心到脸边缘比脸宽
    RTC = generate_specific(tip_of_chin, length_of_face)  # 唇中心到下巴比脸长
    RJ1 = generate_specific(jaw_dist_01, width_of_face)  # 下唇线比脸宽
    RJ2 = generate_specific(jaw_dist_02, width_of_face)  # 下巴上中线比脸宽
    RJ3 = generate_specific(jaw_dist_03, width_of_face)  # 下巴下中线线比脸宽
    RJ4 = generate_specific(jaw_dist_04, width_of_face)  # 下巴底线比脸宽

    # 【5】eyebrow
    width_l_eyebrow = generate_distance(landmark_x[17], landmark_y[17], landmark_x[21], landmark_y[21])  # 左眉宽
    width_R_eyebrow = generate_distance(landmark_x[22], landmark_y[22], landmark_x[26], landmark_y[26])  # 右眉宽
    L_eyebrow_x, L_eyebrow_y = generate_median(landmark_x[17], landmark_y[17], landmark_x[21], landmark_y[21])  # 左眉中心
    R_eyebrow_x, R_eyebrow_y = generate_median(landmark_x[22], landmark_y[22], landmark_x[26], landmark_y[26])  # 右眉中心
    height_l_eyebrow = 2 * generate_distance(L_eyebrow_x, L_eyebrow_y, landmark_x[19], landmark_y[19])  # 左眉高
    height_R_eyebrow = 2 * generate_distance(R_eyebrow_x, R_eyebrow_y, landmark_x[24], landmark_y[24])  # 右眉高
    inner_eyebrow_dist = generate_distance(landmark_x[21], landmark_y[21], landmark_x[22], landmark_y[22])  # 眉间距
    outer_eyebrow_dist = generate_distance(landmark_x[17], landmark_y[17], landmark_x[26], landmark_y[26])  # 眉外间距
    width_of_eyebrow = average(width_l_eyebrow, width_R_eyebrow)
    height_of_eyebrow = average(height_l_eyebrow, height_R_eyebrow)
    RUI = generate_specific(inner_eyebrow_dist, outer_eyebrow_dist)  # 眉内间距比外间距
    RBL = generate_specific(height_of_eyebrow, length_of_face)  # 眉毛高比脸长
    RBH = generate_specific(width_of_eyebrow, width_of_face)  # 眉毛宽比脸宽
    RWH = generate_specific(width_of_eyebrow, height_of_eyebrow)  # 眉毛宽高比
    # 【6】mouse
    top_width_of_mouse = generate_distance(landmark_x[48], landmark_y[48], landmark_x[54], landmark_y[54])
    bot_width_of_mouse = generate_distance(landmark_x[60], landmark_y[60], landmark_x[64], landmark_y[64])
    height_of_mouse = generate_distance(landmark_x[51], landmark_y[51], landmark_x[57], landmark_y[57])
    nose_to_mouse = generate_distance(landmark_x[30], landmark_y[30], landmark_x[62], landmark_y[62])
    nose_to_jaw = generate_distance(landmark_x[30], landmark_y[30], landmark_x[8], landmark_y[8])
    mouse_to_jaw = generate_distance(landmark_x[66], landmark_y[66], landmark_x[8], landmark_y[8])
    RWM = generate_specific(bot_width_of_mouse, height_of_mouse)  # 嘴巴宽高比
    RMW = generate_specific(top_width_of_mouse, width_of_face)  # 嘴巴宽比脸宽
    RMH = generate_specific(height_of_mouse, length_of_face)  # 嘴巴高比脸长
    RNM = generate_specific(nose_to_mouse, length_of_face)  # 鼻子到嘴巴比脸长
    RNF = generate_specific(nose_to_jaw, length_of_face)  # 鼻子到下巴比脸长
    RMF = generate_specific(mouse_to_jaw, length_of_face)  # 嘴到下巴比脸长

    feature = [RF, WHR, RHL, REF, REL, REH, RNO, RNL, RHO, RML,
               RTC, RJ1, RJ2, RJ3, RJ4, RUI, RBL, RBH, RWH, RWM,
               RMW, RMH, RNM, RNF, RMF]

    return feature


landmark_path = 'data/landmarks.csv'
landmark_data = pd.read_csv(landmark_path)  # landmark dataset
Student_quantity = landmark_data.shape[0]
print(Student_quantity)  # two table's amount
Student_number = landmark_data['No']  # students' number

feature_name = ['RF', 'WHR', 'RHL', 'REF', 'REL', 'REH', 'RNO', 'RNL', 'RHO', 'RML',
                'RTC', 'RJ1', 'RJ2', 'RJ3', 'RJ4', 'RUI', 'RBL', 'RBH', 'RWH', 'RWM',
                'RMW', 'RMH', 'RNM', 'RNF', 'RMF']

feature_table = DataFrame(columns=feature_name, index=list(range(1, Student_quantity+1)))

for row in range(0, Student_quantity):
    features = [float(i) for i in generate_features(list(landmark_data.iloc[row]))]
    # print(features)
    feature_table.iloc[row] = features

feature_table.insert(0, 'No', value=[str(i) for i in Student_number])
feature_table.to_csv('data/feature_25.csv', sep=',', encoding='utf-8', index=False, float_format='%.4f')
