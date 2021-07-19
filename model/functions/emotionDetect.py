# -*- coding: utf-8 -*-
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from model.util import FaceUtil
import time
# from oldcare.facial import FaceUtil
import numpy as np
import os
import cv2


class CheckEmotion:
    def __init__(self,model_path=None,faceutil=None):
        self.FACIAL_EXPRESSION_TARGET_WIDTH = 28
        self.FACIAL_EXPRESSION_TARGET_HEIGHT = 28
        self.faceutil = FaceUtil()
        faceutil.save_embeddings()
        self.model_path = os.path.join(os.path.dirname(__file__) + '/../models/face_expression.hdf5')
        self.model = load_model(self.model_path)
        self.time_limit = 60

    def check_emotion(self, frame, stranger_time, result_timer):
        face_location_list, identity = self.faceutil.get_face_location_and_name(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stranger = False
        result = []
        index = 0
        current_time = time.time()
        for (right, top,left, bottom) in face_location_list:
            # extract the ROI of the face from the grayscale image,
            # resize it to a fixed 28x28 pixels, and then prepare the
            # ROI for classification via the CNN
            roi = gray[top:bottom, right:left]
            roi = cv2.resize(roi, (self.FACIAL_EXPRESSION_TARGET_WIDTH, self.FACIAL_EXPRESSION_TARGET_HEIGHT))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # determine facial expression
            (neural, smile) = self.model.predict(roi)[0]
            label = "Natural" if neural > smile else "Smile"

            # display the label and bounding box rectangle on the output
            cv2.putText(frame, identity[index]+":"+label, (right, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 0, 255), 2)

            if(identity[index]=='Unknown'):
                if(current_time - stranger_time > self.time_limit):
                    stranger_time = current_time
                    stranger = True
            else:
                id = identity[index].split("_")
                if(id[1]=="1" and label != "Natural"):
                    if(id[0] not in result_timer or current_time-result_timer[id[0]] > self.time_limit):
                        result_timer[id[0]] = current_time
                        result.append([id[0], label])
            index += 1

        return frame, stranger, stranger_time, result, result_timer

# load the face detector cascade and smile detector CNN
