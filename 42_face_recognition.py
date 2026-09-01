import cv2
import face_recognition
import numpy as np
import os

# ==========================================
# 1. 설정 및 이미지 자동 로딩
# ==========================================
path = 'face_imgs'  # 얼굴 사진들이 들어있는 폴더 이름
known_face_encodings = []
known_face_names = []

# 폴더가 없으면 안내 메시지 출력 후 종료 방지용 생성
if not os.path.exists(path):
    os.makedirs(path)
    print(f"매크로: '{path}' 폴더가 없어서 새로 만들었습니다. 이 안에 사진을 넣어주세요.")
    exit() # 사진이 없으므로 프로그램 종료

# 폴더 내의 파일 리스트 가져오기
image_files = os.listdir(path)
print(f"폴더에서 {len(image_files)}개의 파일을 발견했습니다. 학습을 시작합니다...")

for file_name in image_files:
    # 이미지 파일(.jpg, .png 등)인지 확인 (선택사항이지만 안전을 위해)
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # 파일 이름(확장자 제외)을 사람 이름으로 사용
    name = os.path.splitext(file_name)[0]
    
    # 이미지 불러오기
    img_path = os.path.join(path, file_name)
    cur_img = cv2.imread(img_path)
    
    if cur_img is None:
        print(f" [오류] '{file_name}' 파일을 읽을 수 없습니다.")
        continue

    # OpenCV(BGR) -> Face_recognition(RGB) 변환
    rgb_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 인코딩 (얼굴 위치를 못 찾으면 예외 처리)
    encodings = face_recognition.face_encodings(rgb_img)
    
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        print(f" [완료] '{name}' 학습 완료")
    else:
        print(f" [경고] '{file_name}'에서 얼굴을 찾을 수 없습니다. (제외됨)")

print(f"\n총 {len(known_face_names)}명의 얼굴 학습이 완료되었습니다!")
if not known_face_encodings:
    print(f"'{path}' 폴더에 얼굴이 보이는 사진을 넣고 다시 실행해주세요.")
    raise SystemExit(1)
print("웹캠을 시작합니다... (종료하려면 ESC)")

print("데이터 로딩 완료! 웹캠을 시작합니다...")

# 2. 웹캠 설정
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 거울처럼 좌우 반전 시키기 (1은 좌우 반전, 0은 상하 반전)
    frame = cv2.flip(frame, 1)

    # 속도 향상을 위해 프레임 크기를 1/4로 줄여서 처리
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # OpenCV는 BGR, face_recognition은 RGB를 사용하므로 변환
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 3. 현재 프레임에서 얼굴 위치 찾기 & 인코딩
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 4. 저장된 얼굴들과 현재 얼굴 비교 (매칭)
        # tolerance=0.6 : 숫자가 낮을수록 엄격하게 검사 (0.6이 기본값)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown" # 모르는 사람일 경우

        # 가장 유사한 얼굴 찾기 (거리 계산)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # 5. 화면에 결과 그리기
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 아까 1/4로 줄였으므로 다시 4배로 좌표 복구
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 얼굴 박스 그리기
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # 아는 사람이면 초록, 모르면 빨강
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # 이름표 달기
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    cv2.imshow('Face Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC로 종료
        break

cap.release()
cv2.destroyAllWindows()