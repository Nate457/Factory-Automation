import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureTracker:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        hand_model_path = os.path.join(base_dir, 'tasks', 'hand_landmarker.task')
        pose_model_path = os.path.join(base_dir, 'tasks', 'pose_landmarker_full.task')

        base_options_hands = python.BaseOptions(model_asset_path=hand_model_path)
        options_hands = vision.HandLandmarkerOptions(
            base_options=base_options_hands,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hands)

        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (9, 10), (10, 11), (11, 12),           # Middle
            (13, 14), (14, 15), (15, 16),          # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)              # Palm base
        ]
        
        self.ARM_CONNECTIONS = [
            (11, 12),            # Shoulders
            (11, 13), (13, 15),  # Left Arm
            (12, 14), (14, 16)   # Right Arm
        ]

    def _draw_skeleton(self, frame, landmarks, connections, joint_color, bone_color):
        h, w, _ = frame.shape
        points = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, joint_color, cv2.FILLED)
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], bone_color, 2)

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        hand_results = self.hand_landmarker.detect(mp_image)
        pose_results = self.pose_landmarker.detect(mp_image)
        
        if hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                self._draw_skeleton(frame, hand_landmarks, self.HAND_CONNECTIONS, (0, 0, 255), (0, 255, 0))    
        if pose_results.pose_landmarks:
            for pose_landmarks in pose_results.pose_landmarks:
                self._draw_skeleton(frame, pose_landmarks, self.ARM_CONNECTIONS, (255, 0, 0), (255, 255, 0))
                
        return frame, hand_results, pose_results