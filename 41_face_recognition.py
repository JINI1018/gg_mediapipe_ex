import cv2
import face_recognition
import numpy as np

# 1. 내 얼굴 사진 불러오기 및 인코딩 (얼굴 특징 추출)
# 'my_face.jpg' 파일이 같은 폴더에 있어야함
my_image = face_recognition.load_image_file("face_imgs/joy.jpg")
my_encodings = face_recognition.face_encodings(my_image)
if not my_encodings:
    print("등록 사진에서 얼굴을 찾지 못했습니다. 정면 얼굴 사진으로 바꿔주세요.")
    raise SystemExit(1)
my_face_encoding = my_encodings[0]

# 아는 얼굴들의 목록 (여기서는 '나' 한 명만 등록)
known_face_encodings = [my_face_encoding]
known_face_names = ["joykim"] # 내 이름

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