import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmark,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

from mp_utils import Timestamper, ensure_model

# MediaPipe 1.x 부터는 mp.solutions.pose 가 없어지고 Tasks API 를 사용합니다.
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("pose_landmarker_lite.task")),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as pose:
    timestamper = Timestamper()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = pose.detect_for_video(mp_image, timestamper.next())

        for pose_landmarks in results.pose_landmarks:
            # 포즈 랜드마크 그리기
            drawing_utils.draw_landmarks(
                image,
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style())

            # 예: 코의 좌표 가져오기 (특정 부위 좌표 추출 예시)
            # nose = pose_landmarks[PoseLandmark.NOSE]
            # print(f'Nose coordinates: ({nose.x}, {nose.y})')

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
