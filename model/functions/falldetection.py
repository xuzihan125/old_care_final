# -*- coding: utf-8 -*-
"""
摔倒检测模型
"""
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import time

input_video = None


class FallDetection:
    def __init__(self):
        self.output_fall_path = os.path.join(os.path.dirname(__file__) +'/../supervision/fall')
        self.model_path = os.path.join(os.path.dirname(__file__) +'/../models/fall_detection.hdf5')
        # input_video = '../../tests/corridor_01.avi'
        # 加载模型
        self.model = load_model(self.model_path)
        self.time_limit = 180

    # @staticmethod
    def run(self,transport=None):
        transport['fall']=False
        # determine facial expression
        (fall, normal) = self.model.predict(transport['roi'])[0]

        label = "Fall (%.2f)" % fall if fall > normal else "Normal (%.2f)" % normal

        # display the label and bounding box rectangle on the output frame
        cv2.putText(transport['image'], label, (transport['image'].shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        transport['status'] = 0
        if fall > normal:
            if transport['fall_timing'] == 0:  # just start timing
                transport['fall_timing'] = 1
                transport['fall_start_time'] = time.time()
            else:  # alredy started timing
                fall_end_time = time.time()
                difference = fall_end_time - transport['fall_start_time']

                if difference < transport['fall_limit_time']:
                    # print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                    transport['status'] = 1
                else:  # strangers appear
                    # print('[EVENT] %s, 走廊, 有人摔倒!!!' % current_time)
                    transport['status'] = 2
                    c_time = time.time()
                    if c_time - transport['last_time'] > self.time_limit:
                        transport['last_time'] = c_time
                        transport['fall'] = True
                        # 生成摔倒截图
                        cv2.imwrite(os.path.join(self.output_fall_path,
                                                 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                    transport['image'])  # snapshot
                    # insert into database
                    # command = '%s inserting.py --event_desc %s --event_type 3 --event_location %s' % (
                    #     python_path, event_desc, event_location)
                    # p = subprocess.Popen(command, shell=True)

        return transport


if __name__ == '__main__':
    # 摔倒时间模块
    transportation = {'fall_timing': 0, 'fall_start_time': 0, 'fall_limit_time': 1, 'status': 0,
                      'fall':False,'last_time':0,
                      'roi': None, 'image': None}

    # 全局常量
    TARGET_WIDTH = 64
    TARGET_HEIGHT = 64

    # 初始化摄像头
    if not input_video:
        vs = cv2.VideoCapture(0)
        time.sleep(2)
    else:
        vs = cv2.VideoCapture(input_video)
    # 不断循环
    counter = 0
    fallDetection = FallDetection()
    while True:
        counter += 1
        (grabbed, transportation['image']) = vs.read()
        if input_video and not grabbed:
            break

        if not input_video:
            image = cv2.flip(transportation['image'], 1)

        transportation['roi'] = cv2.resize(transportation['image'], (TARGET_WIDTH, TARGET_HEIGHT))
        transportation['roi'] = transportation['roi'].astype("float") / 255.0
        transportation['roi'] = img_to_array(transportation['roi'])
        transportation['roi'] = np.expand_dims(transportation['roi'], axis=0)

        # 调用算法函数并获取结果
        transportation = fallDetection.run(transportation)
        if transportation['fall']:
            print("有人摔倒")
        # # 对结果进行打印
        # if transportation['status'] == 0:
        #     # print('无人摔倒')
        #     pass
        # elif transportation['status'] == 1:
        #     print('检测摔倒，未记录')
        # elif transportation['status'] == 2:
        #     print('检测到摔倒，已记录')

        cv2.imshow('Fall detection', transportation['image'])

        # Press 'ESC' for exiting video
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vs.release()
    cv2.destroyAllWindows()
