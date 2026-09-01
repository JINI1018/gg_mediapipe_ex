# mediapipe 실습
# 실습 환경
```
# 가상환경
uv init --bare
```

# 설치 라이브러리
```
uv add mediapipe opencv-python 
uv add dlib-bin
uv pip install face-recognition --no-deps
```

# 주의사항
- MediaPipe 1.x 부터 `mp.solutions.hands / pose / face_mesh` 가 삭제되었습니다.
  예제들은 모두 Tasks API(`mediapipe.tasks.python.vision`) 로 작성되어 있습니다.
- Tasks API 는 `.task` 모델 파일이 필요합니다. `mp_utils.ensure_model()` 이
  최초 실행 시 `models/` 폴더로 자동 다운로드합니다. (첫 실행에만 인터넷 필요)