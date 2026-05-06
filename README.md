# 작업자 안전모 착용 자동 탐지 시스템

건설 현장 이미지에서 작업자 안전모 착용 여부를 탐지하고, 안전모 미착용 의심 객체가 있으면 경고 메시지와 로그를 남기는 YOLOv8 기반 MVP 앱입니다.

## 첫 번째 목표

1. 이미지 업로드
2. YOLOv8 객체 탐지
3. `helmet`은 안전, `head`는 안전모 미착용 위험으로 분류
4. 미착용 의심 작업자에게 경고 메시지 표시
5. `runs/alerts.csv`에 이벤트 로그 저장

앱은 보조로 사람 탐지도 함께 사용합니다. 사람의 상단 영역에 안전모가 없으면 `no_helmet_person`으로 경고하므로 일반 캡이나 모자도 안전모가 아닌 것으로 처리할 수 있습니다.

## 실행 방법

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## 모델 연결

처음에는 `models/best.pt`가 없어도 기본 YOLOv8n 모델로 앱이 실행됩니다. 단, 기본 모델은 안전모 전용 모델이 아니므로 실제 안전모 판별을 하려면 공개 안전모 데이터셋으로 YOLOv8을 학습한 뒤 결과 모델을 아래 위치에 넣습니다.

```text
models/best.pt
```

추천 데이터셋:

- Roboflow Hard Hat Workers Dataset
- Kaggle Safety Helmet Detection
- SHWD Safety Helmet Wearing Dataset

## 학습 방법

현재 프로젝트는 Harvard Dataverse의 Hardhat 원본 데이터셋을 Pascal VOC XML에서 YOLOv8 형식으로 변환해 학습할 수 있습니다.

```powershell
python scripts/prepare_yolo_dataset.py
python scripts/train_yolov8.py
```

CPU 환경에서는 시간이 오래 걸리므로 기본값은 작은 실습용 subset과 3 epochs입니다. 더 제대로 학습하려면 GPU 환경에서 아래처럼 늘립니다.

```powershell
python scripts/prepare_yolo_dataset.py --max-train 0 --max-val 0
python scripts/train_yolov8.py --epochs 50 --batch 16 --device 0
```

## 다음 단계

- 이미지 업로드 안전모 탐지
- 동영상 업로드 안전모 탐지
- 웹캠 스냅샷 안전모 탐지
- YOLOv8 Pose 기반 사람 자세선 표시
- 영상/웹캠 입력 추가
- Faster R-CNN 비교 모델 추가
- Meta-CNN 기반 소량 현장 데이터 적응 실험 설계

## 현재 지원 입력

- 이미지: `jpg`, `jpeg`, `png`
- 동영상: `mp4`, `mov`, `avi`
- 웹캠: 브라우저 카메라 스냅샷

실시간 웹캠 스트리밍은 다음 단계에서 `streamlit-webrtc`를 붙여 확장할 수 있습니다.
