# -*- coding: utf-8 -*-

"""
禁止区域检测主程序
摄像头对准围墙那一侧

用法：
python checkingfence.py
python checkingfence.py --filename tests/yard_01.mp4
"""

# import the necessary packages
from model.track import CentroidTracker
from model.track import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import os

input_video = None
# input_video = '../../tests/yard_01.mp4'



class CheckFence:
    def __init__(self):
        self.prototxt_file_path =  os.path.join(os.path.dirname(__file__) + '/../models/mobilenet_ssd/MobileNetSSD_deploy.prototxt')
        self.model_file_path = os.path.join(os.path.dirname(__file__) + '/../models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel')
        self.output_fence_path = os.path.join(os.path.dirname(__file__) + '/../supervision/fence')

        # 全局变量
        # Contains the Caffe deep learning model files.
        # We’ll be using a MobileNet Single Shot Detector (SSD),
        # “Single Shot Detectors for object detection”.
        self.skip_frames = 30  # of skip frames between detections

        # 超参数
        # minimum probability to filter weak detections
        self.minimum_confidence = 0.80

        # 物体识别模型能识别的物体（21种）
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair",
                   "cow", "diningtable", "dog", "horse", "motorbike",
                   "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor"]

        # 加载物体识别模型
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_file_path, self.model_file_path)
        self.time_limit = 60

    def checkFence(self,transport):
        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        transport['intrub'] = False
        transport['status'] = "Waiting"
        rects = []
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if transport['totalFrames'] % self.skip_frames == 0:
            # set the status and initialize our new set of object trackers
            transport['status'] = "Detecting"
            transport['trackers'] = []
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(transport['frame'], 0.007843, (transport['W'], transport['H']), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > self.minimum_confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])
                    # if the class label is not a person, ignore it
                    if self.CLASSES[idx] != "person":
                        continue
                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array(
                        [transport['W'], transport['H'], transport['W'], transport['H']])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(transport['rgb'], rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    transport['trackers'].append(tracker)

        else:
            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            # loop over the trackers
            for tracker in transport['trackers']:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                transport['status'] = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(transport['rgb'])
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # draw a rectangle around the people
                cv2.rectangle(transport['frame'], (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(transport['frame'], (0, transport['H'] // 2), (transport['W'], transport['H'] // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = transport['ct'].update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = transport['trackableObjects'].get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < transport['H'] // 2:
                        transport['totalUp'] += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > transport['H'] // 2:
                        # transport['totalDown'] += 1
                        # to.counted = True
                        #
                        # current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                        #                              time.localtime(time.time()))
                        c_time = time.time()
                        # print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % current_time)
                        if c_time - transport['intrub_time'] > self.time_limit:
                            transport['intrub'] = True
                            transport['intrub_time'] = c_time
                            cv2.imwrite(
                                os.path.join(self.output_fence_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                transport['frame'])  # snapshot

                        # # insert into database
                        # command = '%s inserting.py --event_desc %s --event_type 4 --event_location %s' % (
                        # python_path, event_desc, event_location)
                        # p = subprocess.Popen(command, shell=True)

                    # store the trackable object in our dictionary
            transport['trackableObjects'][objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(transport['frame'], text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(transport['frame'], (centroid[0], centroid[1]), 4,
                       (0, 255, 0), -1)
        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            # ("Up", totalUp),
            ("Down", transport['totalDown']),
            ("Status", transport['status']),
        ]

        # loop over the info transport and draw them on our frame
        # for (i, (w, v)) in enumerate(info):
        #     text = "{}: {}".format(w, v)
        #     cv2.putText(transportation['frame'], text, (10, transportation['H'] - ((i * 20) + 20)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return transport
