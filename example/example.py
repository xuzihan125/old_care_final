#coding=gbk
import os
import time
import cv2
import imutils
from model.functions.dlibPoint import FaceRecord
from imutils import paths
from model.functions.emotionDetect import CheckEmotion
from model.functions.identify import Identify
from model.util.faceutildlib1 import FaceUtil
from model.track import CentroidTracker
from imutils.video import FPS
from model.functions.checkfence import CheckFence
from model.functions.checkvolunteeractivity import CheckVolunteerActivity
from keras.preprocessing.image import img_to_array
import numpy
from model.Model.Detection.Utils import ResizePadding
from model.functions.falldetectionbone import FallDetectionBone
from model.functions.FullEmotion import EmotionDetect


def record():
    # 前置参数
    # 视频来源。需要请自定义
    cap = None

    # 用户id
    userid = "123_1"
    path = "../model/images/%s/"%(userid)
    if not os.path.exists(path):
        os.mkdir(path)

    # 操作顺序及所需照片数
    operation_process = [
        ['face_front', 10],
        ["mouth_open", 10],
        # ["eye_close",10],
        ["face_up", 10],
        # ["face_down", 10],
        ["face_right", 10],
        ["face_left", 10],
    ]

    faceRecord = FaceRecord()

    # 视频来源
    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(0, 640)  # set Width (the first parameter is property_id)
        cap.set(1, 480)  # set Height
        time.sleep(2)
        self_use = True
    else:
        self_use = False

    for (type, count) in operation_process:
        i = 1
        while (i <= count):
            ret, img = cap.read()
            img, check = faceRecord.face_image_process(img, path, photo_name=type + "_" + str(i), display_type="rect",
                                            detect_pos=type)
            if check:
                i = i + 1
            cv2.imshow('img', img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break

    if self_use:
        cap.release()
    cv2.destroyAllWindows()

def check_identity():
    time_dict = {}

    # 初始化摄像头
    vs = cv2.VideoCapture(0)
    time.sleep(2)

    identify = Identify()
    # identify.update()

    # 不断循环
    while True:
        # grab the current frame
        (grabbed, frame) = vs.read()
        frame = cv2.flip(frame, 1)

        # resize the frame, convert it to grayscale, and then clone the
        # original frame so we can draw on it later in the program
        # frame = imutils.resize(frame, width=600)

        frame, ids = identify.check_face(frame)

        # show our detected faces along with smiling/not smiling labels
        cv2.imshow("Face Recognition", frame)

        # Press 'ESC' for exiting video
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()

def emotion():
    # starting video streaming
    transportation = {
        'frame': None, 'stranger': False, 'stranger_time': 0, 'result': [], 'result_timer': {}
    }
    # cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)
    emotionDetect = EmotionDetect()
    while True:

        transportation['frame'] = video_capture.read()[1]
        transportation = emotionDetect.checkEmotion(transportation)
        cv2.imshow('window_frame', transportation['frame'])
        if transportation['stranger']:
            print("陌生人")
        for result in transportation['result']:
            print(result[0]+result[1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    # camera = cv2.VideoCapture(0)
    # time.sleep(2)
    #
    # checkEmotion = CheckEmotion()
    #
    # stranger_time = 0
    # result_timer = {}
    #
    # while True:
    #     (grabbed, frame) = camera.read()
    #     frame = cv2.flip(frame, 1)
    #     # frame = imutils.resize(frame, width=600)
    #
    #     frame,stranger,stranger_time,results,result_time = checkEmotion.check_emotion(frame,stranger_time,result_timer)
    #     if stranger:
    #         print("陌生人")
    #     for result in results:
    #         print(result[0]+result[1])
    #     # show our detected faces along with smiling/not smiling labels
    #     cv2.imshow("Facial Expression Detect", frame)
    #
    #
    #     # Press 'ESC' for exiting video
    #     k = cv2.waitKey(100) & 0xff
    #     if k == 27:
    #         break
    #
    # # cleanup the camera and close any open windows
    # camera.release()
    # cv2.destroyAllWindows()

def checkFence():
    input_video = None
    transportation = {'frame': None, 'status': 0, 'H': None, 'W': None,
                      'intrub': False, 'intrub_time': 0,
                      'ct': CentroidTracker(maxDisappeared=40, maxDistance=50),
                      'trackers': [], 'trackableObjects': {}, 'totalFrames': 0, 'totalDown': 0, 'totalUp': 0,
                      'fps': FPS().start(), 'rgb': None}

    # if a video path was not supplied, grab a reference to the webcam
    if not input_video:
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)
        time.sleep(2)
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(input_video)

    checkFence = CheckFence()
    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        grabbed, transportation['frame'] = vs.read()

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if input_video and not grabbed:
            break

        if not input_video:
            transportation['frame'] = cv2.flip(transportation['frame'], 1)

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        transportation['frame'] = imutils.resize(transportation['frame'], width=500)
        transportation['rgb'] = cv2.cvtColor(transportation['frame'], cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if transportation['W'] is None or transportation['H'] is None:
            (transportation['H'], transportation['W']) = transportation['frame'].shape[:2]

        transportation = checkFence.checkFence(transport=transportation)
        if transportation['intrub']:
            print("有人入侵")
        # show the output frame
        cv2.imshow("Prohibited Area", transportation['frame'])

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        transportation['totalFrames'] += 1
        transportation['fps'].update()

    # stop the timer and display FPS information
    transportation['fps'].stop()

    # close any open windows
    vs.release()
    cv2.destroyAllWindows()

def checkReact():
    input_video = os.path.join(os.path.dirname(__file__) + '/../model/tests/desk_01.mp4')
    transportation = {'frame': None, 'status': 0, 'inter_id': 0, 'camera_turned': 0, 'camera_adjust': 'middle',
                      'camera_angle': 0, 'interact_time': {}, 'interact_list': []}
    # pixel_per_metric = None

    # 全局常量
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480

    # 初始化摄像头
    if not input_video:
        vs = cv2.VideoCapture(0)
        time.sleep(2)
    else:
        vs = cv2.VideoCapture(input_video)

    # 不断循环
    counter = 0
    checkVolunteerActivity = CheckVolunteerActivity(VIDEO_WIDTH,VIDEO_HEIGHT)
    while True:
        counter += 1
        camera_turned = 0
        # grab the current frame
        (grabbed, transportation['frame']) = vs.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if input_video and not grabbed:
            break

        if not input_video:
            transportation['frame'] = cv2.flip(transportation['frame'], 1)

        transportation['frame'] = imutils.resize(transportation['frame'],
                                                 width=VIDEO_WIDTH,
                                                 height=VIDEO_HEIGHT)  # 压缩，为了加快识别速度
        photo_count = 0
        transportation = checkVolunteerActivity.run(transportation)
        # 对结果进行打印
        # 打印相机状态
        # if transportation['camera_turned'] == 1:
        #     print('%d-摄像头需要 turn %s %d 度' % (
        #         counter, transportation['camera_adjust'], transportation['camera_angle']))
        # elif transportation['camera_turned'] == 0:
        #     print('%d-有义工恰好在可见范围内，摄像头不需要转动' % counter)
        # else:
        #     pass
        # 打印接触状态
        if len(transportation['interact_list']) != 0:
            for (old, worker) in transportation['interact_list']:
                print("%s,%s" % (old, worker))

        # show our detected faces along with smiling/not smiling labels
        cv2.imshow("Checking Volunteer's Activities", transportation['frame'])

        # Press 'ESC' for exiting video
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()

def checkFall():
    # output_fall_path = 'Model/saves'
    transportation = {'status': 0, 'frame': None, 'fall_flag': 0,
                      'fall':False,'last_fall':0}

    # 初始化摄像头
    vs = cv2.VideoCapture(0)
    time.sleep(2)
    resize_fn = ResizePadding(384, 384)
    # 不断循环
    counter = 0
    fallDetectionBone = FallDetectionBone(device='cpu')
    while True:
        transportation['status'] = 0
        counter += 1
        camera_turned = 0
        # grab the current frame
        (grabbed, transportation['frame']) = vs.read()

        transportation['frame'] = cv2.flip(transportation['frame'], 1)

        transportation['frame'] = resize_fn(transportation['frame'])
        transportation['frame'] = cv2.cvtColor(transportation['frame'], cv2.COLOR_BGR2RGB)

        transportation = fallDetectionBone.run(transportation)

        if transportation['fall']:
            print('有人摔倒了')
            # # 生成摔倒截图
            # cv2.imwrite(os.path.join(output_fall_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
            #             transportation['frame'])

        cv2.imshow('frame', transportation['frame'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()

    # input_video = None
    # # 摔倒时间模块
    # transportation = {'fall_timing': 0, 'fall_start_time': 0, 'fall_limit_time': 1, 'status': 0,
    #                   'roi': None, 'image': None}
    #
    # # 全局常量
    # TARGET_WIDTH = 64
    # TARGET_HEIGHT = 64
    #
    # # 初始化摄像头
    # if not input_video:
    #     vs = cv2.VideoCapture(0)
    #     time.sleep(2)
    # else:
    #     vs = cv2.VideoCapture(input_video)
    # # 不断循环
    # counter = 0
    #
    # fallDetection = FallDetection()
    # while True:
    #     counter += 1
    #     (grabbed, transportation['image']) = vs.read()
    #     if input_video and not grabbed:
    #         break
    #
    #     if not input_video:
    #         image = cv2.flip(transportation['image'], 1)
    #
    #     transportation['roi'] = cv2.resize(transportation['image'], (TARGET_WIDTH, TARGET_HEIGHT))
    #     transportation['roi'] = transportation['roi'].astype("float") / 255.0
    #     transportation['roi'] = img_to_array(transportation['roi'])
    #     transportation['roi'] = numpy.expand_dims(transportation['roi'], axis=0)
    #
    #     # 调用算法函数并获取结果
    #     transportation = fallDetection.run(transportation)
    #     # 对结果进行打印
    #     if transportation['status'] == 0:
    #         # print('无人摔倒')
    #         pass
    #     elif transportation['status'] == 1:
    #         print('检测摔倒，未记录')
    #     elif transportation['status'] == 2:
    #         print('检测到摔倒，已记录')
    #
    #     cv2.imshow('Fall detection', transportation['image'])
    #
    #     # Press 'ESC' for exiting video
    #     k = cv2.waitKey(1) & 0xff
    #     if k == 27:
    #         break
    #
    # vs.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # record()
    # check_identity()
    emotion()
    # checkFence()
    # checkReact()
    # checkFall()