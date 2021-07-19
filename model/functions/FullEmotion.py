from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import os
import time

from model.utils.datasets import get_labels
from model.utils.inference import detect_faces
from model.utils.inference import draw_text
from model.utils.inference import draw_bounding_box
from model.utils.inference import apply_offsets
from model.utils.inference import load_detection_model
from model.utils.preprocessor import preprocess_input
from model.util import FaceUtil

class EmotionDetect:
    def __init__(self):
        # parameters for loading data and images
        self.detection_model_path = os.path.join(os.path.dirname(__file__) + '/../models/detection_models/haarcascade_frontalface_default.xml')
        self.emotion_model_path = os.path.join(os.path.dirname(__file__) + '/../models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
        self.emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)

        # loading models
        self.face_detection = load_detection_model(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        self.emotion_window = []

        self.faceUtil = FaceUtil()
        self.time_limit = 180

    def checkEmotion(self,transport=None):
        transport['stranger'] = False
        transport['result'] = []

        faces, identity = self.faceUtil.get_face_location_and_name(transport['frame'])
        gray_image = cv2.cvtColor(transport['frame'], cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(transport['frame'], cv2.COLOR_BGR2RGB)
        # faces = detect_faces(self.face_detection, gray_image)
        current_time = time.time()
        for index, face_coordinates in enumerate(faces):
            (x1, y1, x2, y2) = face_coordinates
            face_coordinates = (x1,y1,x2-x1,y2-y1)
            # x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue


            context = ''
            if (identity[index] != 'Unknown'):
                id = identity[index].split("_")
                if (id[1] == "1"):
                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = self.emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = self.emotion_labels[emotion_label_arg]
                    self.emotion_window.append(emotion_text)

                    if len(self.emotion_window) > self.frame_window:
                        self.emotion_window.pop(0)
                    try:
                        emotion_mode = mode(self.emotion_window)
                    except:
                        continue

                    if emotion_text == 'angry':
                        color = emotion_probability * np.asarray((255, 0, 0))
                    elif emotion_text == 'sad':
                        color = emotion_probability * np.asarray((0, 0, 255))
                    elif emotion_text == 'happy':
                        color = emotion_probability * np.asarray((255, 255, 0))
                    elif emotion_text == 'surprise':
                        color = emotion_probability * np.asarray((0, 255, 255))
                    elif emotion_text == 'disgust':
                        color = emotion_probability * np.asarray((255, 0, 255))
                    elif emotion_text == 'fear':
                        color = emotion_probability * np.asarray((255, 255, 255))
                    else:
                        color = emotion_probability * np.asarray((0, 255, 0))
                    color = color.astype(int)
                    color = color.tolist()
                    context = "%s : %s"%(identity[index],emotion_text)
                    if (emotion_text!="neutral" and (id[0] not in transport['result_timer'] or current_time - transport['result_timer'][id[0]] > self.time_limit)):
                        transport['result_timer'][id[0]] = current_time
                        transport['result'].append([id[0], emotion_text])
                else:
                    context = "%s" % (identity[index])
                    color = np.asarray((0, 255, 0))
                    color = color.astype(int)
                    color = color.tolist()
            else:
                context = "%s" % (identity[index])
                color = np.asarray((255, 255, 255))
                color = color.astype(int)
                color = color.tolist()
                if (current_time - transport['stranger_time'] > self.time_limit):
                    transport['stranger_time'] = current_time
                    transport['stranger'] = True

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, context,
                      color, 0, -45, 1, 1)

        transport['frame'] = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return transport