from model.util import FaceUtil
import cv2
import os
from imutils import paths

class Identify:
    def __init__(self,model_path=None):
        self.facial_recognition_model_path = os.path.join(os.path.dirname(__file__) + '/../models/face_recognition_hog.pickle')
        self.faceUtil = FaceUtil(self.facial_recognition_model_path)

    def check_face(self,frame):
        face_location_list, ids = self.faceUtil.get_face_location_and_name(frame)

        # loop over the face bounding boxes
        for ((left, top, right, bottom), name) in zip(
                face_location_list,
                ids):
            # display label and bounding box rectangle on the output frame
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 0, 255), 2)

        return frame, ids

    def update(self):
        images = os.path.join(os.path.dirname(__file__) + "/../images")
        encoding_file_path = os.path.join(os.path.dirname(__file__) + '/../models/face_recognition_hog.pickle')
        self.faceUtil.save_embeddings(list(paths.list_images(images)),encoding_file_path)