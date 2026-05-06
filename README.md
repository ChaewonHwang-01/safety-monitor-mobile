# Safety Monitor Mobile

건설 현장 작업자의 안전모 착용 여부를 이미지, 동영상, 모바일 앱 입력으로 분석하는 멀티모달 안전 모니터링 프로젝트입니다. YOLOv8 기반 PPE 탐지 모델과 보조 headwear 분류기를 결합해 안전모 착용, 일반 캡/모자 착용, 안전모 미착용 상태를 구분하고, 위험 이벤트를 시각/텍스트/음성 경고와 로그로 제공합니다.

## Demo

### Photo Analysis

[음성 포함 사진 분석 데모 영상](docs/assets/readme/photo.mp4)

| 입력 및 분석 | 분석 결과 | 위험 상세 |
|---|---|---|
| ![Photo training](docs/assets/readme/photo_training.png) | ![Photo result](docs/assets/readme/photo_result.png) | ![Photo result info](docs/assets/readme/photo_result_info.png) |

### Video Analysis

[음성 포함 동영상 분석 데모 영상](docs/assets/readme/video.mp4)

| 동영상 분석 | 결과 미리보기 | 이벤트 요약 |
|---|---|---|
| ![Video result](docs/assets/readme/video_result.png) | ![Video2 result](docs/assets/readme/video2_result.png) | ![Video2 result info](docs/assets/readme/video2_result_info.png) |

### Dashboard and Feedback

| 누적 대시보드 | 기간 기반 대시보드 | 피드백 저장 |
|---|---|---|
| ![Dashboard](docs/assets/readme/dashboard.png) | ![Dashboard detail](docs/assets/readme/dashboard2.png) | ![Feedback](docs/assets/readme/photo_feedback.png) |

| 피드백 반영 예시 | 동영상 피드백 | 동영상 학습 데이터 |
|---|---|---|
| ![Feedback ok](docs/assets/readme/photo_feedback_ok.png) | ![Video feedback](docs/assets/readme/video_feedback_new.png) | ![Video2 training](docs/assets/readme/video2_training.png) |

## Key Features

- 이미지 분석: 건설 현장 사진에서 안전모 착용 여부를 탐지합니다.
- 동영상 분석: 업로드한 영상의 일부 프레임을 분석하고, 연속 프레임 탐지를 하나의 위험 이벤트로 묶습니다.
- 모바일 앱: Expo 기반 모바일 앱에서 촬영/갤러리 업로드/영상 분석을 수행합니다.
- 멀티모달 경고: 빨간 바운딩 박스, 텍스트 메시지, 선택적 음성 안내를 함께 제공합니다.
- 위험 이벤트 로그: 분석 시간, 입력 유형, 위험 개수, 메시지를 CSV 로그로 저장합니다.
- 누적 위험 대시보드: 사용자가 선택한 기간의 위험 분석 기록과 위험 이벤트를 집계합니다.
- 오탐/미탐 피드백: 잘못된 탐지 결과를 올바른 라벨로 저장해 추후 보조 분류기 재학습 데이터로 활용할 수 있습니다.
- 일반 캡/모자 분리: 안전모가 아닌 일반 캡/모자는 별도 위험 유형으로 분리합니다.

## Detection Policy

이 프로젝트는 현장 안전 관점에서 다음 기준으로 판정합니다.

| 판정 | 의미 | 처리 |
|---|---|---|
| `helmet` | 정상적인 안전모 착용 | 안전 |
| `cap_hat` | 일반 캡, 모자, 후드 등 안전모가 아닌 착용물 | 위험 |
| `no_helmet` | 머리에 아무것도 착용하지 않은 상태 | 위험 |
| `no_helmet_person` | 사람은 감지됐지만 상단 영역에서 안전모가 확인되지 않은 보조 판정 | 위험 |

앱 화면에서는 사용자 이해를 위해 위험 유형을 크게 `일반 캡/모자`와 `안전모 미착용` 중심으로 보여주도록 설계했습니다.

## System Architecture

```mermaid
flowchart LR
    A["Image / Video / Mobile Camera"] --> B["FastAPI Analysis Server"]
    B --> C["YOLOv8 PPE Detector"]
    B --> D["Person Fallback Detector"]
    C --> E["Headwear Classifier"]
    D --> E
    E --> F["Risk Policy Engine"]
    F --> G["Annotated Result"]
    F --> H["Alert Log CSV"]
    H --> I["Cumulative Dashboard"]
    G --> J["Streamlit Web App"]
    G --> K["Expo Mobile App"]
```

## Project Structure

```text
.
├── app.py                         # Streamlit web app
├── api_server.py                  # FastAPI backend for mobile app
├── desktop_app.py                 # Desktop wrapper experiment
├── src/
│   ├── detector.py                # YOLOv8 detection and headwear verification
│   ├── video.py                   # Video frame sampling and event grouping
│   ├── alert.py                   # Risk message rules
│   └── logger.py                  # CSV event logging
├── scripts/
│   ├── prepare_yolo_dataset.py
│   ├── train_yolov8.py
│   ├── prepare_headwear_classifier_dataset.py
│   ├── train_headwear_classifier.py
│   └── training_status.py
├── mobile_app/
│   ├── App.tsx
│   └── src/api.ts
└── docs/assets/readme/            # README demo images and videos
```

## Setup

### 1. Python Environment

```powershell
pip install -r requirements.txt
```

### 2. Model Files

Model weights are not included in this repository because they are large generated artifacts. Place trained weights in the following paths:

```text
models/best.pt
models/headwear_cls.pt
```

- `models/best.pt`: YOLOv8 safety helmet detection model
- `models/headwear_cls.pt`: second-stage classifier for `helmet`, `cap_hat`, `bare_head`

If these files are missing, the app may fall back to default YOLO models, but real safety helmet performance will be limited.

## Run

### Streamlit Web App

```powershell
streamlit run app.py
```

### FastAPI Analysis Server

```powershell
.\run_api_server.bat
```

The API server runs on:

```text
http://127.0.0.1:8000
```

For a physical phone, use the laptop's Wi-Fi IP address in the mobile app, for example:

```text
http://192.168.x.x:8000
```

### Expo Mobile App

```powershell
cd mobile_app
npm install
npx expo start
```

Then scan the QR code with Expo Go.

## Training

### YOLOv8 Helmet Detector

```powershell
python scripts/prepare_yolo_dataset.py
python scripts/train_yolov8.py
```

For longer training on GPU:

```powershell
python scripts/train_yolov8.py --epochs 70 --batch 16 --device 0
```

Training progress can be checked with:

```powershell
python scripts/training_status.py --run runs/train/hardhat-plus-yolov8n-70e --epochs 70
```

### Headwear Classifier

The second-stage classifier is used to reduce confusion between hard hats, caps, and bare heads.

```powershell
python scripts/prepare_headwear_classifier_dataset.py
python scripts/train_headwear_classifier.py
```

Manual feedback crops saved through the app can be reused as additional classifier data.

## 피드백 루프

피드백 화면은 단순히 사람이 결과를 수정하는 기능이 아니라, 추후 모델 개선을 위한 학습 데이터를 모으는 구조로 설계했습니다.

1. 모델이 잘못된 라벨을 예측합니다.
2. 사용자가 올바른 라벨을 선택합니다: `helmet`, `cap_hat`, `bare_head`.
3. 해당 탐지 영역 crop이 `data/headwear_cls/manual/` 아래에 저장됩니다.
4. 저장된 피드백 데이터를 포함해 보조 분류기 데이터셋을 다시 구성합니다.
5. headwear classifier를 재학습한 뒤 `models/headwear_cls.pt`로 교체합니다.

이 과정을 통해 앱 테스트 중 발견되는 오탐/미탐 사례를 다시 학습 데이터로 반영하는 작은 active learning 흐름을 만들 수 있습니다.

## 한계점

- 본 프로젝트는 프로토타입이므로 실제 안전 관리 시스템으로 사용하려면 현장 데이터 기반 검증이 추가로 필요합니다.
- 성능은 데이터셋 품질, 촬영 각도, 조명, 거리, 가려짐 정도에 큰 영향을 받습니다.
- 일반 캡/모자, 후드, 부분적으로 보이는 안전모, 저해상도 영상 프레임에서는 여전히 오탐이나 미탐이 발생할 수 있습니다.
- 모바일 앱은 현재 모델을 휴대폰 내부에서 직접 실행하지 않고, 로컬 FastAPI 서버로 이미지를 전송해 분석합니다.
- 모델 가중치와 데이터셋은 용량 문제로 GitHub 저장소에 포함하지 않았기 때문에 별도로 준비해야 합니다.

## 향후 개선 방향

- `helmet`, `cap_hat`, `bare_head` 샘플을 더 균형 있게 수집해 headwear classifier를 개선합니다.
- TensorFlow Lite, Core ML, ONNX Runtime Mobile 등을 활용해 온디바이스 추론을 추가합니다.
- 동영상 분석에서 이벤트 단위 confidence smoothing을 적용해 순간적인 오탐을 줄입니다.
- 선택한 기간의 안전 리포트를 PDF/CSV로 내보내는 기능을 추가합니다.
- 현장 관리자와 안전 관리자를 위한 역할별 대시보드 화면을 설계합니다.
- 안전모뿐 아니라 안전조끼, 장갑, 안전벨트, 위험 자세 등으로 PPE/행동 분석 범위를 확장합니다.

## Tech Stack

- Python
- YOLOv8 / Ultralytics
- OpenCV
- FastAPI
- Streamlit
- React Native / Expo
- Pandas

## Repository

GitHub: [safety-monitor-mobile](https://github.com/ChaewonHwang-01/safety-monitor-mobile)
