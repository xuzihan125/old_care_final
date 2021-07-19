import os

import cv2
import time

import torch
import numpy as np


from model.Model.Tools.DetectorLoader import TinyYOLOv3_onecls

from model.Model.Tools.PoseEstimateLoader import SPPE_FastPose

from model.Model.Track.Tracker import Detection, Tracker
from model.Model.Tools.ActionsEstLoader import TSSTG




def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

class FallDetectionBone:
    def __init__(self,device='cuda'):
        # DETECTION MODEL.
        self.detect_model = TinyYOLOv3_onecls(384, device='cpu')
        # POSE MODEL.
        self.inp_pose = '224x160'.split('x')
        self.inp_pose = (int(self.inp_pose[0]), int(self.inp_pose[1]))
        self.pose_model = SPPE_FastPose('resnet50', self.inp_pose[0], self.inp_pose[1], device='cpu')
        # Actions Estimate.
        self.action_model = TSSTG(device='cpu')
        self.tracker = Tracker(max_age=30, n_init=3)
        self.time_limit = 180




    # @staticmethod
    def run(self,transport=None):
        transport['fall'] = False
        # Detect humans bbox in the frame with detector model.
        detected = self.detect_model.detect(transport['frame'], need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        self.tracker.predict()
        # Merge two source of predicted bbox together.
        for track in self.tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = self.pose_model.predict(transport['frame'], detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        self.tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed():
                transport['status'] = 0
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                transport['fall_flag'] = transport['fall_flag'] + 1
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = self.action_model.predict(pts, transport['frame'].shape[:2])
                action_name = self.action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down' and transport['fall_flag'] >= 30:
                    c_time = time.time()
                    if c_time - transport['last_fall'] > self.time_limit:
                        transport['fall'] = True
                        transport['last_fall'] = c_time
                    transport['status'] = 1
                    transport['fall_flag'] = 0
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                # 显示骨骼
                # frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(transport['frame'], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                transport['frame'] = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                                 0.4, clr, 1)

        # Show Frame.
        transport['frame'] = cv2.resize(transport['frame'], (0, 0), fx=2., fy=2.)
        transport['frame'] = transport['frame'][:, :, ::-1]

        return transport