# MediaPipe Windows 실습 환경 설정

MediaPipe와 OpenCV로 웹캠을 사용하는 실습 프로젝트입니다.

## Windows에서 별도로 실행하는 이유

WSL은 Linux 개발 환경에는 편리하지만 Windows에 연결된 웹캠과 GUI 창을 바로 사용하는 실습에서는 장치 연결, 화면 출력, 권한 설정이 복잡해질 수 있습니다. 웹캠 입력과 OpenCV 창 출력을 안정적으로 확인하기 위해 이 프로젝트는 WSL 환경과 분리하고 Windows PowerShell에서 실행합니다.

## 1. 저장소 준비

[Git for Windows](https://git-scm.com/download/win)를 설치한 뒤 PowerShell을 새로 열고 저장소를 복제합니다.

```powershell
git clone <원본-저장소-URL> C:\Users\Admin\gg1th_mediapipe_ex_win
cd C:\Users\Admin\gg1th_mediapipe_ex_win
```

이후 명령은 모두 이 프로젝트 폴더에서 실행합니다.

## 2. uv와 Python 버전 설정

이 가이드는 Windows에 [`uv`](https://docs.astral.sh/uv/getting-started/installation/)가 설치되어 있고 PowerShell에서 `uv` 명령을 사용할 수 있다는 전제로 작성했습니다.

`pyproject.toml`에서 Python 요구 버전을 3.12로 맞춥니다.

```toml
[project]
requires-python = ">=3.12,<3.13"
```

이전에 다른 Python 버전으로 만든 가상환경이 있다면 먼저 삭제합니다.

```powershell
Remove-Item -Recurse -Force .venv
```

`.venv`가 없다는 오류가 나오면 그대로 다음 단계로 진행하면 됩니다.

Python 3.12를 설치하고 프로젝트 버전으로 고정한 뒤 의존성을 동기화합니다.

```powershell
uv python install 3.12
uv python pin 3.12
uv sync
uv run python --version
```

마지막 명령의 결과가 `Python 3.12.x`인지 확인합니다.

## 3. 주요 라이브러리 설치

프로젝트 의존성이 아직 등록되지 않았다면 다음과 같이 설치할 수 있습니다.

```powershell
uv add mediapipe opencv-python
uv add dlib-bin
uv pip install face-recognition-models==0.3.0
uv pip install face-recognition --no-deps
```

### Windows application control / OS error 4551

`face-recognition-models==0.3.0` 설치 중 Windows Application Control 정책 때문에 OS error 4551이 발생할 수 있습니다. `uv`의 기본 캐시 위치에서 파일 실행이 차단되는 경우 프로젝트 내부의 별도 캐시 폴더를 지정한 뒤 다시 설치해 봅니다.

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv pip install face-recognition-models==0.3.0
```

이 설정은 현재 PowerShell 세션에만 적용됩니다. 조직이나 학교에서 관리하는 PC라면 보안 정책 자체가 설치를 막을 수 있으므로, 우회가 되지 않을 때는 관리자에게 허용 여부를 확인해야 합니다.

## 4. 실행 확인

스크립트는 `uv run`으로 실행합니다.

```powershell
uv run python <실습파일.py>
```

Windows 카메라 권한에서 데스크톱 앱의 카메라 접근이 허용되어 있어야 합니다. 다른 프로그램이 웹캠을 사용 중이면 종료한 뒤 다시 실행합니다.

## 5. 새 GitHub 저장소에 연결

새 GitHub 저장소를 만든 뒤 기존 `origin`을 새 주소로 변경하고 결과를 확인합니다.

```powershell
git remote set-url origin https://github.com/<GitHub-ID>/<저장소명>.git
git remote -v
```

변경 사항을 커밋하고 `main` 브랜치를 푸시합니다.

```powershell
git add .
git commit -m "Windows MediaPipe 실습 환경 설정"
git branch -M main
git push -u origin main
```

처음 푸시할 때 `please complete authentication in your browser`가 표시되면 열린 브라우저에서 GitHub 로그인을 완료합니다. `Authentication Succeeded`가 나오면 인증은 정상적으로 끝난 것입니다.

GitHub에서 저장소를 만들 때 README, `.gitignore`, LICENSE 등을 함께 생성했다면 원격 저장소에 초기 커밋이 있어 `fetch first` 오류가 날 수 있습니다. 원격 내용을 버려도 되는 새 실습 저장소임을 확인한 경우에만 다음 명령으로 로컬 내용을 올립니다.

```powershell
git push -u origin main --force
```

`--force`는 원격 `main`의 기존 커밋을 덮어쓸 수 있습니다. 공동 작업 저장소나 보존할 내용이 있는 저장소에서는 사용하지 않습니다.

## MediaPipe 참고사항

- MediaPipe 1.x에서는 예전 `mp.solutions.hands`, `pose`, `face_mesh` 방식 대신 Tasks API(`mediapipe.tasks.python.vision`)를 사용합니다.
- Tasks API 실행에는 `.task` 모델 파일이 필요합니다.
- 이 프로젝트의 `mp_utils.ensure_model()`은 최초 실행 시 모델을 `models/` 폴더에 내려받습니다.
- 공식 문서: <https://developers.google.com/edge/mediapipe/solutions/guide>
