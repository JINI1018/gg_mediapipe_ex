import cv2
import mediapipe as mp
import numpy as np
import math
import os

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from mp_utils import Timestamper, ensure_model

# === 설정 ===
glasses_path = "./images/glasses3.png" # 안경 이미지 파일명
resize_scale = 1.6           # 안경 크기 조절 비율 (눈 사이 거리의 몇 배인지)
y_offset = 6              # 안경을 콧대보다 얼마나 내릴/올릴지 (양수면 아래로)

# === 초기화 ===
# MediaPipe 1.x 부터는 mp.solutions.face_mesh 가 없어지고 Tasks API 를 사용합니다.
face_mesh = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
))
timestamper = Timestamper()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# === 유틸리티 함수 ===

# 1. 이미지 로드 (투명도 유지)
def load_glasses(path):
    if not os.path.exists(path):
        # 파일 없으면 임시로 보라색 박스 안경 생성
        print(f"이미지({path})가 없어서 임시 안경을 생성합니다.")
        img = np.zeros((100, 300, 4), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (130, 80), (255, 0, 255, 255), 5) # 왼쪽 알
        cv2.rectangle(img, (170, 20), (280, 80), (255, 0, 255, 255), 5) # 오른쪽 알
        cv2.line(img, (130, 50), (170, 50), (255, 0, 255, 255), 5)      # 브릿지
        return img
        
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("이미지 읽기 실패")
        return None
    return img

# 2. 이미지 회전 함수
def rotate_image(image, angle):
    # 이미지 중심을 기준으로 회전
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 매트릭스 생성
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전 시 이미지가 잘리지 않도록 캔버스 크기 재계산 (선택 사항이지만 추천)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 회전 실행 (투명도 채널 유지를 위해 borderValue 투명 설정)
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

# 3. 투명 이미지 겹치기 함수 (이전 예제와 동일)
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, ov_c = overlay.shape

    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background

    bg_x1 = max(x, 0)
    bg_y1 = max(y, 0)
    bg_x2 = min(x + ov_w, bg_w)
    bg_y2 = min(y + ov_h, bg_h)

    ov_x1 = max(0, -x)
    ov_y1 = max(0, -y)
    ov_x2 = ov_x1 + (bg_x2 - bg_x1)
    ov_y2 = ov_y1 + (bg_y2 - bg_y1)

    overlay_cropped = overlay[ov_y1:ov_y2, ov_x1:ov_x2]

    # 알파 블렌딩
    alpha_mask = overlay_cropped[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_mask
    roi = background[bg_y1:bg_y2, bg_x1:bg_x2]

    for c in range(3):
        roi[:, :, c] = (alpha_mask * overlay_cropped[:, :, c] +
                        alpha_inv * roi[:, :, c])
    background[bg_y1:bg_y2, bg_x1:bg_x2] = roi
    return background

# 안경 이미지 불러오기
glasses_img_orig = load_glasses(glasses_path)

print("안경 피팅 시작! (종료: ESC)")

while True:
    success, img = cap.read()
    if not success: break
    
    # 거울 모드
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = face_mesh.detect_for_video(mp_image, timestamper.next())

    if results.face_landmarks:
        # Tasks API 결과는 얼굴마다 NormalizedLandmark 리스트를 바로 담고 있습니다.
        for face_landmarks in results.face_landmarks:
            # === 핵심 로직 ===
            
            # 1. 기준점 좌표 가져오기 (왼쪽 눈꼬리: 33, 오른쪽 눈꼬리: 263)
            # MediaPipe Face Mesh Landmark Map 참조
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            
            # 화면 좌표로 변환
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            
            # 2. 안경 각도 계산 (Roll)
            # 두 눈을 잇는 선의 기울기를 계산 (아크탄젠트)
            dY = ry - ly
            dX = rx - lx
            angle = math.degrees(math.atan2(dY, dX)) # 라디안 -> 도(degree) 변환
            
            # 3. 안경 크기 계산
            # 두 눈 사이의 거리(유클리드 거리) 구하기
            eye_dist = math.sqrt((dX ** 2) + (dY ** 2))
            
            # 안경 너비를 눈 거리의 N배로 설정 (resize_scale)
            glasses_width = int(eye_dist * resize_scale)
            
            # 비율 유지하며 높이 계산
            orig_h, orig_w = glasses_img_orig.shape[:2]
            glasses_height = int(glasses_width * (orig_h / orig_w))
            
            # 안경 이미지 리사이즈
            glasses_resized = cv2.resize(glasses_img_orig, (glasses_width, glasses_height))
            
            # 4. 안경 회전 (고개 기울기 반영)
            # OpenCV 회전 함수는 반시계 방향이 양수이므로 -angle 적용
            glasses_rotated = rotate_image(glasses_resized, -angle)
            
            # 5. 위치 잡기 (Center)
            # 두 눈의 정중앙 지점 계산
            center_x = (lx + rx) // 2
            center_y = (ly + ry) // 2
            
            # 콧대 위치 보정 (안경 중심을 콧대로 맞추기)
            # 회전된 이미지의 크기
            gh, gw = glasses_rotated.shape[:2]
            
            # 그릴 위치 (좌상단 좌표)
            final_x = center_x - (gw // 2)
            final_y = center_y - (gh // 2) + y_offset

            # 6. 화면에 합성
            img = overlay_transparent(img, glasses_rotated, final_x, final_y)
            
            # (디버깅용) 눈 좌표 표시
            # cv2.circle(img, (lx, ly), 5, (0, 0, 255), -1)
            # cv2.circle(img, (rx, ry), 5, (0, 0, 255), -1)

    cv2.imshow("Virtual Glasses Fitting", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
