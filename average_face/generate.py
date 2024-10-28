# ---* Feature point location *--
import os
import pandas as pd

LANDMARK_PATH = './landmark'
student_landmark = os.listdir(LANDMARK_PATH)
student_num = len(student_landmark)

# 2014148207
# 2014250832
# 2015082104
# 2016083113
landmark_table = pd.DataFrame(columns=list(range(1, 137)), index=list(range(1, student_num+1)))

student_number = []
for stu in student_landmark:

    file_name = LANDMARK_PATH + '/' + stu

    lines = open(file_name).readlines()
    with open(file_name, 'w') as fp:
        for s in lines:
            fp.write(s.replace(' ', '\n'))

    with open(file_name, 'r') as fp:
        lines = fp.read().splitlines()
        mark_list = []
        for line in lines:
            mark_list.append(line)
        fp.close()
        # print(len(feature_list))
        if len(mark_list) == 0:
            print(line)
        else:
            pass
    landmark_list = [int(i) for i in mark_list]
    landmark_table.iloc[student_landmark.index(stu)] = landmark_list
    student_num = str(os.path.splitext(stu)[0])
    student_number.append(student_num)

landmark_table.insert(0, 'No', value=student_number)
landmark_table.to_csv('./data/landmarks.csv', sep=',', encoding='utf-8', index=False)
