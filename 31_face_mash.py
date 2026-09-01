import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    RunningMode,
    drawing_utils,
)

from mp_utils import Timestamper, ensure_model

# MediaPipe 1.x 부터는 mp.solutions.face_mesh 가 없어지고 Tasks API 를 사용합니다.
# face_landmarker.task 모델은 눈동자(iris)를 포함한 478개 랜드마크를 제공합니다.
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as face_mesh:
    timestamper = Timestamper()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = face_mesh.detect_for_video(mp_image, timestamper.next())

        for face_landmarks in results.face_landmarks:
            drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,  # 얼굴 그물망
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    thickness=1, circle_radius=1, color=(255, 255, 255)))

        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
