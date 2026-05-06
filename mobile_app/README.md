# Safety Monitor Mobile App

YOLOv8 안전모 탐지 백엔드(`api_server.py`)와 연결되는 Expo 기반 모바일 앱입니다.

## 1. 백엔드 실행

프로젝트 루트에서 실행합니다.

```powershell
pip install -r requirements.txt
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

또는 루트의 `run_api_server.bat`를 더블클릭합니다.

## 2. 노트북 IP 확인

실제 휴대폰에서 앱을 실행할 경우 `127.0.0.1`이 아니라 노트북의 Wi-Fi IP를 사용해야 합니다.

```powershell
ipconfig
```

예시:

```text
http://192.168.0.25:8000
```

앱 상단의 `분석 서버 주소`에 이 주소를 입력합니다.

## 3. 모바일 앱 실행

```powershell
cd mobile_app
npm install
.\run_mobile_app.bat
```

Expo Go 앱으로 QR 코드를 스캔하면 휴대폰에서 실행할 수 있습니다.

## 현재 포함 기능

- 카메라 촬영 또는 갤러리 이미지 선택
- YOLOv8 백엔드 분석 요청
- 바운딩 박스 결과 이미지 표시
- 위험 이벤트 상세 표시
- 누적 위험 대시보드 조회
- 오탐/미탐 피드백 샘플 저장

## 피드백 데이터 흐름

앱에서 저장한 피드백은 루트의 `data/headwear_cls/manual/<label>`에 crop 이미지로 저장됩니다.
일정량이 쌓이면 보조 분류기 재학습에 사용합니다.

```powershell
python scripts\prepare_headwear_classifier_dataset.py --max-per-class 300
python scripts\train_headwear_classifier.py --epochs 30 --imgsz 224 --batch 16 --device cpu --name headwear-cls-feedback
```
