import dlib
from imutils import paths
import face_recognition
import os
import cv2
import pickle

class FaceUtil:

    def __init__(self,encoding_data_path=None):
        self.imgPath = os.path.join(os.path.dirname(__file__) + "/../images")
        self.identity_model = os.path.join(os.path.dirname(__file__) + '/../models/face_recognition_hog.pickle')
        self.detector = dlib.get_frontal_face_detector()
        self.encoding_file_path = os.path.join(os.path.dirname(__file__) + '/../models/face_recognition_hog.pickle')
        self.data = pickle.loads(open(self.encoding_file_path,"rb").read())
        self.tolerance = 0.5
        self.detection_method = "hog"

    def get_face_location(self, image):
        face_location_list = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = face_recognition.face_locations(
            gray, number_of_times_to_upsample=1,
            model=self.detection_method)
        for (top, right, bottom, left) in dets:
            face_location_list.append((left, top, right, bottom))
        return face_location_list

    def get_face_location_list(self, image):
        face_location_list = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = face_recognition.face_locations(
            gray, number_of_times_to_upsample=1,
            model=self.detection_method)
        for (top, right, bottom, left) in dets:
            face_location_list.append((left, top, right, bottom))
        return face_location_list

    def get_main_face_location(self, image):
        dets = self.get_face_location(image)
        if (len(dets) == 0):
            return None
        elif (len(dets) == 1):
            return dets[0]
        face_area = [(det[0]-det[2])*(det[1]-det[3]) for det in dets]
        max = face_area[0]
        max_index = 0
        for i in range(1, len(face_area)):
            if (face_area[i] > max):
                max = face_area[i]
                max_index = i
        return dets[max_index]

    def get_face_location_and_name(self,image):
        boxes = self.get_face_location_list(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("111",image)
        # for box in boxes:
        #     cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),color=(0,0,255))
        # cv2.imshow("tempo",image)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to
            # our known encodings
            matches = face_recognition.compare_faces(
                self.data["encodings"], encoding,
                tolerance=self.tolerance)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then
                # initialize a dictionary to count the total number
                #  of times each face was matched
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count
                # for each recognized face face
                for i in matched_idxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest
                # number of votes (note: in the event of an unlikely
                # tie Python will select first entry in the
                # dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        face_location_list = []
        for ((right,top,  left, bottom)) in boxes:
            face_location_list.append((right, top,left,  bottom))

        return face_location_list, names

    def update_identity_model(self):
        image_paths = list(paths.list_images(self.imgPath))
        output_encoding_file_path = self.identity_model
        warning = ''
        known_encodings = []
        known_names = []

        for (i, image_path) in enumerate(image_paths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}"
                  .format(i + 1, len(image_paths)))
            name = image_path.split(os.path.sep)[-2]  # person name

            # load the input image and convert it from
            # RGB (OpenCV ordering) to dlib ordering (RGB)
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(
                rgb, model=self.detection_method)

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            if len(encodings) != 1:
                os.remove(image_path)
                warning += '[WARNING] detected multiple faces in %s.' % (name)
                # warning += ' This file is deleted.\n' %(
                #         len(encodings), image_path)
                continue

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names
                # and encodings
                known_encodings.append(encoding)
                known_names.append(name)

        # dump the facial encodings + names to disk
        self.data = {"encodings": known_encodings, "names": known_names}
        f = open(output_encoding_file_path, "wb")
        f.write(pickle.dumps(self.data))
        f.close()

        if warning:
            print(warning)

        return




