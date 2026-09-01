"""MediaPipe Tasks API 예제 공용 유틸리티.

MediaPipe 1.x 부터는 `mp.solutions.*` (Hands / Pose / FaceMesh) 가 삭제되고
Tasks API 로 통합되었습니다. Tasks API 는 모델 파일(.task)을 직접 지정해야 하므로,
여기서 모델을 한 번만 내려받아 `models/` 폴더에 캐시해 둡니다.
"""

import os
import urllib.request

BASE_URL = "https://storage.googleapis.com/mediapipe-models"

MODEL_URLS = {
    "hand_landmarker.task":
        f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task":
        f"{BASE_URL}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker.task":
        f"{BASE_URL}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def ensure_model(file_name):
    """모델 파일 경로를 돌려준다. 없으면 내려받는다."""
    if file_name not in MODEL_URLS:
        raise KeyError(f"등록되지 않은 모델입니다: {file_name}")

    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"모델 다운로드 중... {file_name}")
        urllib.request.urlretrieve(MODEL_URLS[file_name], path)
        print(f"다운로드 완료 -> {path}")
    return path


class Timestamper:
    """VIDEO 모드에 필요한, 항상 증가하는 밀리초 타임스탬프 생성기."""

    def __init__(self):
        import time
        self._time = time.time
        self._start = self._time()
        self._last = -1

    def next(self):
        ms = max(int((self._time() - self._start) * 1000), self._last + 1)
        self._last = ms
        return ms
