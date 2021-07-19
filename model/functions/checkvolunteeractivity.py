# -*- coding: utf-8 -*-
"""
义工是否和老人有互动主程序

用法：
python checkingvolunteeractivity.py
python checkingvolunteeractivity.py --filename tests/desk_01.mp4
"""

from model.util import FaceUtil
from scipy.spatial import distance as dist
from model.utils import fileassistant
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import os
import imutils
import numpy as np

# # input_video = None
# input_video = '../../tests/desk_01.mp4'
# output_activity_path = '../../supervision/activity'
# model_path = '../../models/face_recognition_hog.pickle'
# people_info_path = '../../info/people_info.csv'
# # 加载模型
# faceutil = FaceUtil(model_path)
# # 得到 ID->姓名的map 、 ID->职位类型的map
# id_card_to_name, id_card_to_type = fileassistant.get_people_info(people_info_path)
#
# FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
# ANGLE = 20
# ACTUAL_DISTANCE_LIMIT = 100  # cm


class CheckVolunteerActivity:
    def __init__(self,VIDEO_WIDTH,VIDEO_HEIGHT):
        # self.input_video =
        self.output_activity_path = os.path.join(os.path.dirname(__file__) + '/../supervision/activity')
        self.model_path = os.path.join(os.path.dirname(__file__) + '/../models/face_recognition_hog.pickle')
        self.people_info_path = os.path.join(os.path.dirname(__file__) + '/../info/people_info.csv')
        # 加载模型
        self.faceutil = FaceUtil(self.model_path)
        # 得到 ID->姓名的map 、 ID->职位类型的map
        self.id_card_to_name, self.id_card_to_type = fileassistant.get_people_info(self.people_info_path)

        self.FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
        self.ANGLE = 20
        self.ACTUAL_DISTANCE_LIMIT = 100  # cm
        self.time_limit = 300
        self.VIDEO_WIDTH=VIDEO_WIDTH
        self.VIDEO_HEIGHT=VIDEO_HEIGHT

    # @staticmethod
    def run(self,transport=None):
        transport['interact_list'] = []
        # 将相机状态重置
        transport['camera_turned'] = 0
        # 将接触状态重置
        transport['status'] = 0
        face_location_list, names = self.faceutil.get_face_location_and_name(transport['frame'])

        # 得到画面的四分之一位置和四分之三位置，并垂直划线
        one_sixth_image_center = (int(self.VIDEO_WIDTH / 6), int(self.VIDEO_HEIGHT / 6))
        five_sixth_image_center = (int(self.VIDEO_WIDTH / 6 * 5),
                                   int(self.VIDEO_HEIGHT / 6 * 5))

        cv2.line(transport['frame'], (one_sixth_image_center[0], 0),
                 (one_sixth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)
        cv2.line(transport['frame'], (five_sixth_image_center[0], 0),
                 (five_sixth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)

        people_type_list = list(set([self.id_card_to_type[i] for i in names]))

        volunteer_name_direction_dict = {}
        volunteer_centroids = []
        worker_name_list = []
        old_people_centroids = []
        old_people_name = []

        # loop over the face bounding boxes
        for ((left, top, right, bottom), name) in zip(face_location_list, names):  # 处理单个人

            person_type = self.id_card_to_type[name]
            # 将人脸框出来
            rectangle_color = (0, 0, 255)
            if person_type == 'old_people':
                rectangle_color = (0, 0, 128)
            elif person_type == 'employee':
                rectangle_color = (255, 0, 0)
            elif person_type == 'volunteer':
                rectangle_color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(transport['frame'], (left, top), (right, bottom),
                          rectangle_color, 2)

            if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
                continue

            if person_type == 'volunteer' or person_type == 'employee':  # 如果检测到有义工存在
                worker_name_list.append(name)
                # 获得义工位置
                volunteer_face_center = (int((right + left) / 2),
                                         int((top + bottom) / 2))
                volunteer_centroids.append(volunteer_face_center)

                cv2.circle(transport['frame'],
                           (volunteer_face_center[0], volunteer_face_center[1]),
                           8, (255, 0, 0), -1)

                adjust_direction = ''
                # face locates too left, servo need to turn right,
                # so that face turn right as well
                if volunteer_face_center[0] < one_sixth_image_center[0]:
                    adjust_direction = 'right'
                elif volunteer_face_center[0] > five_sixth_image_center[0]:
                    adjust_direction = 'left'

                volunteer_name_direction_dict[name] = adjust_direction

            elif person_type == 'old_people':  # 如果没有发现义工
                old_people_face_center = (int((right + left) / 2),
                                          int((top + bottom) / 2))
                old_people_centroids.append(old_people_face_center)
                old_people_name.append(name)

                cv2.circle(transport['frame'],
                           (old_people_face_center[0], old_people_face_center[1]),
                           4, (0, 255, 0), -1)
            else:
                pass

            # 人脸识别和表情识别都结束后，把表情和人名写上 (同时处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(transport['frame'], cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            final_label = self.id_card_to_name[name]
            draw.text((left, top - 30), final_label,
                      font=ImageFont.truetype('arial.ttf', 40),
                      fill=(255, 0, 0))  # linux
            # 转换回OpenCV格式
            transport['frame'] = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # 义工追踪逻辑
        if 'volunteer' in people_type_list or 'employee' in people_type_list:
            volunteer_adjust_direction_list = list(volunteer_name_direction_dict.values())
            if '' in volunteer_adjust_direction_list:  # 有的义工恰好在范围内，所以不需要调整舵机
                transport['camera_turned'] = 0
                # print('%d-有义工恰好在可见范围内，摄像头不需要转动' % counter)
            else:
                transport['camera_adjust'] = volunteer_adjust_direction_list[0]
                transport['camera_angle'] = self.ANGLE
                transport['camera_turned'] = 1
                # print('%d-摄像头需要 turn %s %d 度' % (counter, adjust_direction, ANGLE))

        c_time = time.time()
        # 在义工和老人之间划线
        if transport['camera_turned'] == 0:
            for i_index, i in enumerate(volunteer_centroids):
                for j_index, j in enumerate(old_people_centroids):
                    pixel_distance = dist.euclidean(i, j)
                    face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(
                        face_location_list)
                    pixel_per_metric = face_pixel_width / self.FACE_ACTUAL_WIDTH
                    actual_distance = pixel_distance / pixel_per_metric

                    if actual_distance < self.ACTUAL_DISTANCE_LIMIT:
                        cv2.line(transport['frame'], (int(i[0]), int(i[1])),
                                 (int(j[0]), int(j[1])), (255, 0, 255), 2)
                        label = 'distance: %dcm' % actual_distance
                        cv2.putText(transport['frame'], label, (transport['frame'].shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                        # current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                        #                              time.localtime(time.time()))
                        # event_desc = '%s正在与义工交互' % (id_card_to_name[old_people_name[j_index]])
                        # event_location = '房间桌子'
                        last_time = transport['interact_time'][self.id_card_to_name[old_people_name[j_index]]+'+'+self.id_card_to_name[worker_name_list[i_index]]]
                        if c_time - last_time > self.time_limit:
                            transport['interact_time'][
                                self.id_card_to_name[old_people_name[j_index]] + '+' + self.id_card_to_name[
                                    worker_name_list[i_index]]] = c_time

                            transport['interact_list'].append([ self.id_card_to_name[old_people_name[j_index]],self.id_card_to_name[worker_name_list[i_index]]])
                        transport['status'] = 1
                        # transport['inter_id'] = self.id_card_to_name[old_people_name[j_index]]
                        # print('[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (
                        #     current_time, id_card_to_name[old_people_name[j_index]]))

                        # cv2.imwrite(
                        #     os.path.join(self.output_activity_path,
                        #                  'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                        #     transport['frame'])  # snapshot

                        # # insert into database
                        # command = '%s inserting.py --event_desc %s --event_type 1 --event_location %s --old_people_id %d' %(python_path, event_desc, event_location, int(name))
                        # p = subprocess.Popen(command, shell=True)

        return transport



