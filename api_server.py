from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.alert import reason_for_detection
from src.detector import DetectionResult, SafetyHelmetDetector
from src.logger import AlertLogger
from src.video import process_video


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best.pt"
LOG_PATH = ROOT / "runs" / "alerts.csv"
FEEDBACK_DIR = ROOT / "data" / "headwear_cls" / "manual"
UPLOAD_DIR = ROOT / "runs" / "mobile_uploads"
VIDEO_DIR = ROOT / "runs" / "mobile_videos"

RISK_TYPE_LABELS = {
    "cap_hat": "일반 캡",
    "no_helmet": "안전모 미착용",
}

RISK_TYPE_DESCRIPTIONS = {
    "cap_hat": "머리에 착용물이 있지만 안전모가 아니므로 일반 캡으로 분류합니다.",
    "no_helmet": "머리에 아무것도 착용하지 않은 상태로 판단합니다.",
}

NO_HELMET_ALIASES = {"bare_head", "no_helmet", "no_helmet_person"}


app = FastAPI(title="Construction Safety Monitor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = SafetyHelmetDetector(model_path=MODEL_PATH)
logger = AlertLogger(LOG_PATH)


class FeedbackRequest(BaseModel):
    image_base64: str
    bbox: list[int]
    correct_label: str


def image_to_base64(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_from_base64(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def position_label(detection: dict, image_size: tuple[int, int]) -> str:
    width, height = image_size
    x_center = (detection["x1"] + detection["x2"]) / 2
    y_center = (detection["y1"] + detection["y2"]) / 2
    horizontal = "좌측" if x_center < width / 3 else "우측" if x_center > width * 2 / 3 else "중앙"
    vertical = "상단" if y_center < height / 3 else "하단" if y_center > height * 2 / 3 else "중단"
    return f"{vertical} {horizontal}"


def build_message(alert_count: int) -> str:
    if alert_count == 0:
        return "현재 입력에서는 안전모 미착용 위험이 감지되지 않았습니다."
    return f"안전모 미착용 위험 {alert_count}건이 감지되었습니다. 안전모를 착용하세요."


def normalized_risk_class(detection_class: str) -> str:
    if detection_class in NO_HELMET_ALIASES:
        return "no_helmet"
    return detection_class


def risk_label(detection_class: str) -> str:
    return RISK_TYPE_LABELS.get(normalized_risk_class(detection_class), detection_class)


def risk_description(detection_class: str) -> str:
    if detection_class == "cap_hat":
        return RISK_TYPE_DESCRIPTIONS["cap_hat"]
    normalized = normalized_risk_class(detection_class)
    return RISK_TYPE_DESCRIPTIONS.get(normalized, reason_for_detection(detection_class))


def detection_to_feedback_item(detection: dict, index: int, image_size: tuple[int, int]) -> dict:
    detection_class = detection["class"]
    return {
        "id": f"D{index:02d}",
        "class": detection_class,
        "label": risk_label(detection_class),
        "confidence": detection["confidence"],
        "status": detection["status"],
        "position": position_label(detection, image_size),
        "bbox": [detection["x1"], detection["y1"], detection["x2"], detection["y2"]],
        "reason": risk_description(detection_class),
    }


def risk_details(result: DetectionResult) -> list[dict]:
    details = []
    image_size = result.annotated_image.size
    risk_index = 1
    for detection in result.detections:
        if detection["status"] != "risk":
            continue
        detection_class = detection["class"]
        normalized_class = normalized_risk_class(detection_class)
        details.append(
            {
                "risk_id": f"R{risk_index:02d}",
                "class": normalized_class,
                "raw_class": detection_class,
                "label": risk_label(detection_class),
                "confidence": detection["confidence"],
                "position": position_label(detection, image_size),
                "bbox": [detection["x1"], detection["y1"], detection["x2"], detection["y2"]],
                "reason": risk_description(detection_class),
                "recommended_action": "작업자에게 안전모 착용을 안내하고 보호구 착용 후 작업을 재개하도록 확인",
            }
        )
        risk_index += 1
    return details


def normalize_video_events(events: list[dict]) -> list[dict]:
    normalized_events = []
    for event in events:
        raw_class = str(event.get("class") or "unknown")
        normalized_class = normalized_risk_class(raw_class)
        normalized_events.append(
            {
                **event,
                "class": normalized_class,
                "raw_class": raw_class,
                "label": risk_label(raw_class),
                "reason": risk_description(raw_class),
            }
        )
    return normalized_events


def filter_events_by_date(events: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if events.empty or "timestamp" not in events.columns:
        return events

    frame = events.copy()
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame[parsed.notna()].copy()
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")

    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start):
            frame = frame[parsed >= start]
            parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    if end_date:
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end):
            frame = frame[parsed < end + pd.Timedelta(days=1)]

    return frame


def summarize_events(events: pd.DataFrame, start_date: str | None = None, end_date: str | None = None) -> dict:
    events = filter_events_by_date(events, start_date, end_date)
    if events.empty:
        return {
            "total_logs": 0,
            "total_risks": 0,
            "by_risk_type": [],
            "date_range": {"start_date": start_date, "end_date": end_date},
            "definitions": {
                "total_logs": "선택한 기간 안에서 위험이 1건 이상 있었던 분석 기록 수입니다.",
                "total_risks": "선택한 기간 안에서 각 기록에 저장된 실제 위험 이벤트 수의 합계입니다.",
            },
        }

    frame = events.copy()
    frame["alert_count"] = pd.to_numeric(frame.get("alert_count", 0), errors="coerce").fillna(0).astype(int)

    risk_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        raw_details = row.get("risk_details_json", "")
        details = []
        if isinstance(raw_details, str) and raw_details.strip():
            try:
                details = json.loads(raw_details)
            except json.JSONDecodeError:
                details = []
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            risk_class = normalized_risk_class(str(detail.get("class") or "unknown"))
            risk_counts[risk_class] = risk_counts.get(risk_class, 0) + int(detail.get("count") or 1)

    by_risk_type = [
        {
            "risk_type": key,
            "label": RISK_TYPE_LABELS.get(key, key),
            "description": RISK_TYPE_DESCRIPTIONS.get(key, ""),
            "count": value,
        }
        for key, value in sorted(risk_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "total_logs": len(frame),
        "total_risks": int(frame["alert_count"].sum()),
        "by_risk_type": by_risk_type,
        "date_range": {"start_date": start_date, "end_date": end_date},
        "definitions": {
            "total_logs": "선택한 기간 안에서 위험이 1건 이상 있었던 분석 기록 수입니다.",
            "total_risks": "선택한 기간 안에서 각 기록에 저장된 실제 위험 이벤트 수의 합계입니다.",
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "ok": detector.is_ready,
        "model_path": str(MODEL_PATH),
        "message": detector.status_message(),
    }


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    source_type: str = Form("mobile"),
) -> dict:
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.") from exc

    result = detector.detect(
        image,
        confidence=confidence,
        draw_pose=False,
        risk_confidence=confidence,
        person_fallback_confidence=max(confidence, 0.30),
        min_person_area_ratio=0.004,
        imgsz=960,
    )
    details = risk_details(result)
    feedback_items = [
        detection_to_feedback_item(detection, index, result.annotated_image.size)
        for index, detection in enumerate(result.detections, start=1)
    ]
    message = build_message(result.alert_count)

    if result.alert_count:
        summary = "; ".join(f"{item['risk_id']} {item['position']} {item['class']}" for item in details)
        logger.write_event(
            source_name=file.filename or "mobile_upload",
            source_type=source_type,
            alert_count=result.alert_count,
            total_detections=len(result.detections),
            message=message,
            risk_summary=summary,
            risk_details=details,
            alert_modes=["visual", "text", "voice", "mobile"],
        )

    return {
        "kind": "image",
        "message": message,
        "alert_count": result.alert_count,
        "total_detections": len(result.detections),
        "detections": result.detections,
        "risk_details": details,
        "feedback_items": feedback_items,
        "annotated_image_base64": image_to_base64(result.annotated_image),
        "source_image_base64": image_to_base64(image),
    }


@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    frame_stride: int = Form(3),
    max_frames: int = Form(180),
) -> dict:
    suffix = Path(file.filename or "mobile_video.mp4").suffix or ".mp4"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    input_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    output_path = VIDEO_DIR / f"{input_path.stem}_annotated.mp4"
    input_path.write_bytes(await file.read())

    result = process_video(
        input_path=input_path,
        output_path=output_path,
        detector=detector,
        confidence=confidence,
        draw_pose=False,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )

    video_events = normalize_video_events(result.risk_events)

    if result.risk_event_count:
        details = [
            {
                **event,
                "count": 1,
                "position": "동영상 전체",
                "recommended_action": "위험 구간을 확인하고 안전모 착용 여부를 점검",
            }
            for event in video_events
        ]
        summary = "; ".join(
            f"{item['risk_id']} {item['class']} {item['start_time']}s~{item['end_time']}s"
            for item in details
        )
        logger.write_event(
            source_name=file.filename or "mobile_video",
            source_type="video",
            alert_count=result.risk_event_count,
            total_detections=result.processed_frames,
            message=f"동영상에서 위험 이벤트 {result.risk_event_count}건이 감지되었습니다.",
            risk_summary=summary,
            risk_details=details,
            alert_modes=["visual", "text", "voice", "mobile"],
        )

    return {
        "kind": "video",
        "message": f"동영상에서 위험 이벤트 {result.risk_event_count}건이 감지되었습니다.",
        "alert_count": result.risk_event_count,
        "total_detections": result.processed_frames,
        "risk_frame_count": result.risk_frame_count,
        "raw_alert_count": result.total_alerts,
        "risk_events": video_events,
        "preview_gif_base64": file_to_base64(result.preview_path) if result.preview_path.exists() else "",
    }


@app.get("/events")
def events(start_date: str | None = None, end_date: str | None = None) -> dict:
    return summarize_events(logger.read_events(), start_date=start_date, end_date=end_date)


@app.post("/feedback/headwear")
def save_headwear_feedback(request: FeedbackRequest) -> dict:
    if request.correct_label not in {"helmet", "cap_hat", "bare_head"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 라벨입니다.")
    if len(request.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox는 [x1, y1, x2, y2] 형식이어야 합니다.")

    image = image_from_base64(request.image_base64)
    width, height = image.size
    x1, y1, x2, y2 = request.bbox
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="유효하지 않은 bbox입니다.")

    target_dir = FEEDBACK_DIR / request.correct_label
    target_dir.mkdir(parents=True, exist_ok=True)
    crop = image.crop((x1, y1, x2, y2))
    output_path = target_dir / f"{uuid4().hex}.jpg"
    crop.save(output_path, quality=95)

    return {
        "saved": True,
        "label": request.correct_label,
        "path": str(output_path),
    }
