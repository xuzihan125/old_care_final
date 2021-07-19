# coding=gbk
import dlib
import numpy
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
from model.util.faceutildlib import FaceUtil
import os


class FaceRecord:
    def __init__(self):
        self.set()

    def set(self):
        # 参数定义
        # flag of self self-created img input flow
        self.self_use = False

        # face-point-dect model path
        self.predictor_path = os.path.join(os.path.dirname(__file__) + "/../models/shape_predictor_68_face_landmarks.dat")

        # detector initialize
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # util
        self.faceUtil = FaceUtil()

        # word dict
        self.word_dict = {
            "lack_face": "未检测到人脸",
            "processing": "正在拍照",
            "eye_close": "请闭眼",
            "mouth_open": "请张嘴",
            "face_front":"请向前看",
            "face_up": "请向上看",
            "face_down": "请向下看",
            "face_right": "请向右看",
            "face_left": "请向左看",
            "face_unfront": "请将头摆正"
        }

        # point in use
        self.dict_point = {
            "eye_transvers": [[43, 47], [44, 46], [37, 41], [38, 40]],
            "eye_vertical": [[45, 42], [36, 39]],
            "mouth_transvers": [[49, 59], [50, 58], [52, 56], [53, 55]],
            "mouth_vertical": [[36, 39]],
            "direct": [30, 8, 36, 45, 48, 54]
        }

        self.dict_limit = {
            "eye_limit": [0, 0.2],
            "mouth_limit": [1, 100],
            "front_limit": [-15, 15],
            "direct_limit":[[165,200],[-180,-170],[-30,30]],
            "up_limit": [90, 165],
            "down_limit": [-170, -90],
            "right_limit": [30, 180],
            "left_limit": [-180, -30]
        }

        # 3D point
        self.model_points = numpy.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        self.function_dict = {
            "no": self.checkTemplate,
            "eye_close": self.checkEye,
            "mouth_open": self.checkMouth,
            "face_front":self.checkFront,
            "face_up": self.checkUp,
            "face_down": self.checkDown,
            "face_right": self.checkRight,
            "face_left": self.checkLeft,
        }

    # 获得人脸角度
    def getDirection(self, landmark, size):
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = numpy.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        image_points = numpy.array([
            (landmark.part(a).x, landmark.part(a).y) for a in self.dict_point["direct"]
        ], dtype="double")

        dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion
        # model_points1 = model_points

        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return self.get_euler_angle(rotation_vector)

    # 计算欧拉角
    def get_euler_angle(self, rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        pitch = math.atan2(t0, t1)

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)

        # 单位转换：将弧度转换为度
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)

        return Y, X, Z

    # 关键点校验
    # 校验模板
    def checkTemplate(self, landark, vector):
        return True

    # 眼部校验
    def checkEye(self, landmark, vector):
        transvers = 0
        for (a, b) in self.dict_point["eye_transvers"]:
            transvers = transvers + ((landmark.part(a).x - landmark.part(b).x) ** 2 + (
                        landmark.part(a).y - landmark.part(b).y) ** 2) ** 0.5
        vertical = 0
        for (a, b) in self.dict_point["eye_vertical"]:
            vertical = vertical + ((landmark.part(a).x - landmark.part(b).x) ** 2 + (
                        landmark.part(a).y - landmark.part(b).y) ** 2) ** 0.5
        total = (transvers / len(self.dict_point["eye_transvers"])) / (vertical / len(self.dict_point["eye_vertical"]))

        if total >= self.dict_limit["eye_limit"][0] and total <= self.dict_limit["eye_limit"][1]:
            return True
        else:
            return False

    # 嘴部校验
    def checkMouth(self, landmark, vector=None):
        transvers = 0
        for (a, b) in self.dict_point["mouth_transvers"]:
            transvers = transvers + ((landmark.part(a).x - landmark.part(b).x) ** 2 + (
                    landmark.part(a).y - landmark.part(b).y) ** 2) ** 0.5
        vertical = 0
        for (a, b) in self.dict_point["mouth_vertical"]:
            vertical = vertical + ((landmark.part(a).x - landmark.part(b).x) ** 2 + (
                    landmark.part(a).y - landmark.part(b).y) ** 2) ** 0.5
        total = (transvers / len(self.dict_point["mouth_transvers"])) / (
                    vertical / len(self.dict_point["mouth_vertical"]))
        print(total)
        if total >= self.dict_limit["mouth_limit"][0] and total <= self.dict_limit["mouth_limit"][1]:
            return True
        else:
            return False

    # 面部姿态校验
    def checkFront(self,landmark, vector):
        (y,x,z) = vector
        return (y > self.dict_limit["direct_limit"][0][0] or y < self.dict_limit["direct_limit"][1][1]) and \
               (x > self.dict_limit["direct_limit"][2][0] and x < self.dict_limit["direct_limit"][2][1])

    def checkUp(self, landmark, vector):
        (y, x, z) = vector
        return y > self.dict_limit["up_limit"][0] and y < self.dict_limit["up_limit"][1]

    def checkDown(self, landmark, vector):
        (y, x, z) = vector
        return y > self.dict_limit["down_limit"][0] and y < self.dict_limit["down_limit"][1]

    def checkRight(self, landmark, vector):
        (y, x, z) = vector
        return x > self.dict_limit["right_limit"][0] and x < self.dict_limit["right_limit"][1]

    def checkLeft(self, landmark, vector):
        (y, x, z) = vector
        return x > self.dict_limit["left_limit"][0] and x < self.dict_limit["left_limit"][1]

    def checkFront(self, landmark, vector):
        (y, x, z) = vector
        return z > self.dict_limit["front_limit"][0] and z < self.dict_limit["front_limit"][1]

    # 绘制文字
    def drawText(self, img, text, pos, textSize, color):
        if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(pos, text, color, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    # input the image, out put the processed image
    # photo_name 为照片编号
    # display_type: "no":无需显示 "rect"：方框框出 "point_p"：面部关键点-点 "point_n"：面部关键点-数
    # detect_pos: "no":无需检测 "eye_close":闭眼, "mouth_open":张嘴, "face_up":向上看, "face_down":向下看, "face_right":向右看, "face_left":向左看,
    # 输出：(图片，是否姿势正确)
    def face_image_process(self, img, path, photo_name=1, display_type="no", detect_pos="no"):
        ori_img = img.copy()
        size = img.shape

        if size[0] > 700:
            h = size[0] / 3
            w = size[1] / 3
            im = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
            size = im.shape

        # 检测人脸
        # dets = detector(img, 1)
        det = self.faceUtil.get_main_face_location(img)  # ace(dets)
        if det is None:
            img = self.drawText(img, self.word_dict["lack_face"], (10, 30), 15, (0, 255, 0))
            return img, False

        # 检测姿态并存储
        shape = self.predictor(img, dlib.rectangle(det[0],det[1],det[2],det[3]))
        landmark = shape  # numpy.matrix([[p.x, p.y] for p in shape.parts()])
        vector = self.getDirection(landmark, size)
        if not self.checkFront(landmark, vector):
            img = self.drawText(img, self.word_dict["face_unfront"], (10, 30), 15, (0, 255, 0))
            check = False
        else:
            check = self.function_dict[detect_pos](landmark, vector)
            if check:
                cv2.imwrite(path+"%s.jpg" % (photo_name), ori_img)
                img = self.drawText(img, self.word_dict["processing"], (10, 30), 15, (0, 255, 0))
            else:
                img = self.drawText(img, self.word_dict[detect_pos], (10, 30), 15, (0, 255, 0))

        # 人脸标注
        if (display_type == "rect"):
            cv2.rectangle(img, (det[0],det[1]), (det[2], det[3]), color=(0, 255, 0))
        elif (display_type == "point_p"):
            for point in landmark.parts():
                pos = (point.x, point.y)
                cv2.circle(img, pos, 3, color=(0, 255, 0))
        elif (display_type == "point_n"):
            for idx, point in enumerate(landmark.parts()):
                pos = (point.x, point.y)
                cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=0.3, color=(0, 255, 0))
        return img, check
