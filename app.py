from __future__ import annotations

import html
import base64
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from src.alert import build_alert_message, reason_for_detection
from src.detector import DetectionResult, SafetyHelmetDetector
from src.logger import AlertLogger
from src.video import process_video


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best.pt"
LOG_PATH = ROOT / "runs" / "alerts.csv"
UPLOAD_DIR = ROOT / "runs" / "uploads"
VIDEO_DIR = ROOT / "runs" / "videos"
HEADWEAR_SAMPLE_DIR = ROOT / "data" / "headwear_cls" / "manual"


st.set_page_config(
    page_title="Construction Safety Monitor",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stCameraInput"] video,
    div[data-testid="stCameraInput"] img {
        transform: scaleX(-1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector() -> SafetyHelmetDetector:
    return SafetyHelmetDetector(model_path=MODEL_PATH)


def position_label(detection: dict, image_size: tuple[int, int]) -> str:
    width, height = image_size
    x_center = (detection["x1"] + detection["x2"]) / 2
    y_center = (detection["y1"] + detection["y2"]) / 2

    horizontal = "좌측" if x_center < width / 3 else "우측" if x_center > width * 2 / 3 else "중앙"
    vertical = "상단" if y_center < height / 3 else "하단" if y_center > height * 2 / 3 else "중단"
    return f"{vertical} {horizontal}"


def risk_details(result: DetectionResult) -> list[dict]:
    image_size = result.annotated_image.size
    details = []
    risk_index = 1

    for detection in result.detections:
        if detection["status"] != "risk":
            continue

        details.append(
            {
                "risk_id": f"R{risk_index:02d}",
                "class": detection["class"],
                "confidence": detection["confidence"],
                "position": position_label(detection, image_size),
                "bbox": f"{detection['x1']},{detection['y1']},{detection['x2']},{detection['y2']}",
                "reason": reason_for_detection(detection["class"]),
                "recommended_action": "작업자에게 안전모 착용을 안내하고, 보호구 착용 후 작업을 재개하도록 확인",
            }
        )
        risk_index += 1

    return details


def ppe_policy_summary(result: DetectionResult) -> pd.DataFrame:
    rows = []
    helmet_count = sum(1 for item in result.detections if item["class"] == "helmet")
    no_helmet_count = sum(1 for item in result.detections if item["status"] == "risk")

    rows.append(
        {
            "항목": "안전모",
            "판정": f"{helmet_count}건 감지",
            "정책": "안전 보호구로 인정",
        }
    )
    rows.append(
        {
            "항목": "일반 캡/모자/후드",
            "판정": "별도 안전모로 인정하지 않음",
            "정책": "사람 상단에 helmet이 없으면 no_helmet_person으로 경고",
        }
    )
    rows.append(
        {
            "항목": "미착용 위험",
            "판정": f"{no_helmet_count}건",
            "정책": "시각/텍스트/음성 경고 및 이벤트 로그 저장",
        }
    )
    return pd.DataFrame(rows)


SOURCE_TYPE_LABELS = {
    "image": "이미지",
    "video": "동영상",
    "webcam": "웹캠",
}

RISK_TYPE_LABELS = {
    "cap_hat": "일반 캡/모자",
    "bare_head": "머리 노출",
    "no_helmet": "안전모 미착용",
    "no_helmet_person": "사람 상단 안전모 없음",
}


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_risk_counts(row: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw_details = row.get("risk_details_json", "")

    if isinstance(raw_details, str) and raw_details.strip():
        try:
            details = json.loads(raw_details)
        except json.JSONDecodeError:
            details = []

        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                risk_class = str(detail.get("class") or "unknown")
                count = _safe_int(detail.get("count"), 1)
                counts[risk_class] = counts.get(risk_class, 0) + max(1, count)

    if not counts:
        summary = row.get("risk_summary", "")
        if isinstance(summary, str):
            for risk_class in RISK_TYPE_LABELS:
                if risk_class in summary:
                    counts[risk_class] = counts.get(risk_class, 0) + 1

    if not counts:
        alert_count = _safe_int(row.get("alert_count"), 0)
        if alert_count:
            counts["unknown"] = alert_count

    return counts


def _risk_label(risk_class: str) -> str:
    return RISK_TYPE_LABELS.get(risk_class, risk_class)


def render_alert_dashboard(events: pd.DataFrame) -> None:
    st.subheader("누적 위험 대시보드")

    dashboard = events.copy()
    dashboard["timestamp_dt"] = pd.to_datetime(dashboard.get("timestamp"), errors="coerce")
    dashboard["source_type"] = dashboard.get("source_type", "image").fillna("image")
    dashboard["alert_count"] = dashboard.get("alert_count", 0).apply(_safe_int)
    dashboard["total_detections"] = dashboard.get("total_detections", 0).apply(_safe_int)
    dashboard["risk_counts"] = dashboard.apply(_parse_risk_counts, axis=1)
    dashboard["risk_types"] = dashboard["risk_counts"].apply(lambda counts: sorted(counts.keys()))

    available_source_types = sorted(
        source_type for source_type in dashboard["source_type"].dropna().unique().tolist()
    )
    available_risk_types = sorted(
        {
            risk_type
            for risk_types in dashboard["risk_types"]
            for risk_type in risk_types
        }
    )

    valid_dates = dashboard["timestamp_dt"].dropna()
    if valid_dates.empty:
        min_date = max_date = pd.Timestamp.now().date()
    else:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.0, 1.2])
    with filter_col1:
        selected_period = st.date_input(
            "조회 기간",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with filter_col2:
        selected_sources = st.multiselect(
            "입력 유형",
            available_source_types,
            default=available_source_types,
            format_func=lambda value: SOURCE_TYPE_LABELS.get(value, value),
        )
    with filter_col3:
        selected_risks = st.multiselect(
            "위험 유형",
            available_risk_types,
            default=available_risk_types,
            format_func=_risk_label,
        )

    filtered = dashboard.copy()
    if isinstance(selected_period, tuple) and len(selected_period) == 2:
        start_date, end_date = selected_period
    else:
        start_date = end_date = selected_period
    filtered = filtered[
        (filtered["timestamp_dt"].dt.date >= start_date)
        & (filtered["timestamp_dt"].dt.date <= end_date)
    ]

    if selected_sources:
        filtered = filtered[filtered["source_type"].isin(selected_sources)]

    if selected_risks:
        filtered = filtered[
            filtered["risk_types"].apply(
                lambda risk_types: any(risk_type in selected_risks for risk_type in risk_types)
            )
        ]

    total_risks = int(filtered["alert_count"].sum()) if not filtered.empty else 0
    total_rows = len(filtered)
    total_detections = int(filtered["total_detections"].sum()) if not filtered.empty else 0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("위험 로그", total_rows)
    metric_col2.metric("총 위험 건수", total_risks)
    metric_col3.metric("전체 탐지/처리 수", total_detections)
    metric_col4.metric("입력 유형 수", filtered["source_type"].nunique() if not filtered.empty else 0)

    if filtered.empty:
        st.info("선택한 조건에 해당하는 위험 로그가 없습니다.")
        return

    risk_rows = []
    for counts in filtered["risk_counts"]:
        for risk_class, count in counts.items():
            risk_rows.append(
                {
                    "위험 유형": _risk_label(risk_class),
                    "건수": count,
                }
            )
    risk_summary = (
        pd.DataFrame(risk_rows).groupby("위험 유형", as_index=False)["건수"].sum()
        if risk_rows
        else pd.DataFrame(columns=["위험 유형", "건수"])
    )

    source_summary = (
        filtered.groupby("source_type", as_index=False)
        .agg(로그수=("source_type", "size"), 위험건수=("alert_count", "sum"))
        .assign(입력유형=lambda frame: frame["source_type"].map(lambda value: SOURCE_TYPE_LABELS.get(value, value)))
        [["입력유형", "로그수", "위험건수"]]
    )

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.write("미착용 유형별 건수")
        st.dataframe(risk_summary, use_container_width=True, hide_index=True)
    with summary_col2:
        st.write("입력 종류별 기록")
        st.dataframe(source_summary, use_container_width=True, hide_index=True)

    if not risk_summary.empty:
        st.bar_chart(risk_summary.set_index("위험 유형"))

    display_columns = [
        column
        for column in [
            "timestamp",
            "source",
            "source_type",
            "alert_count",
            "risk_summary",
            "alert_modes",
            "message",
        ]
        if column in filtered.columns
    ]
    display_events = filtered[display_columns].copy()
    if "source_type" in display_events.columns:
        display_events["source_type"] = display_events["source_type"].map(
            lambda value: SOURCE_TYPE_LABELS.get(value, value)
        )
    st.write("필터 적용 로그")
    st.dataframe(display_events, use_container_width=True, hide_index=True)

    with st.expander("상세 JSON 로그 보기"):
        detail_column = "risk_details_json"
        if detail_column in filtered.columns:
            st.dataframe(filtered[["timestamp", "source", detail_column]], use_container_width=True)


def speak_alert(message: str, enabled: bool) -> None:
    if not enabled:
        return

    safe_message = json.dumps(message, ensure_ascii=False)
    components.html(
        f"""
        <script>
        const message = new SpeechSynthesisUtterance({safe_message});
        message.lang = "ko-KR";
        message.rate = 1.0;
        message.pitch = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(message);
        </script>
        """,
        height=0,
    )


def render_gif_preview(preview_path: Path, width: int = 480) -> None:
    encoded = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    components.html(
        f"""
        <img
            src="data:image/gif;base64,{encoded}"
            style="width:{width}px; max-width:100%; border-radius:6px; display:block;"
            alt="분석 결과 미리보기"
        />
        """,
        height=min(760, width * 2),
    )


def crop_detection(image: Image.Image, detection: dict, padding: float = 0.14) -> Image.Image:
    width, height = image.size
    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    return image.crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )
    )


def render_headwear_sample_collector(result: DetectionResult, source_name: str, source_type: str) -> None:
    source_image = result.source_image
    if source_image is None or not result.detections:
        return

    candidate_indexes = [
        index
        for index, detection in enumerate(result.detections)
        if detection["class"] in {"helmet", "cap_hat", "no_helmet", "no_helmet_person"}
    ]
    if not candidate_indexes:
        return

    stable_source = "".join(char if char.isalnum() else "_" for char in f"{source_type}_{source_name}")

    with st.expander("Headwear classifier sample collector"):
        detection_options = {
            (
                f"{index}: {detection['class']} "
                f"det={detection['confidence']} "
                f"check={detection.get('headwear_check', '-')}/"
                f"{detection.get('headwear_confidence', '-')}"
            ): index
            for index, detection in enumerate(result.detections)
            if index in candidate_indexes
        }
        selected_label = st.selectbox(
            "Detection crop to save",
            list(detection_options.keys()),
            key=f"sample_detection_{stable_source}",
        )
        label = st.selectbox(
            "Correct label for this crop",
            ["cap_hat", "helmet", "bare_head"],
            key=f"sample_label_{stable_source}",
        )
        target_dir = HEADWEAR_SAMPLE_DIR / label
        current_count = len(list(target_dir.glob("*.jpg"))) if target_dir.exists() else 0
        selected_index = detection_options[selected_label]
        preview_crop = crop_detection(source_image, result.detections[selected_index])
        st.image(preview_crop, caption="Crop preview", width=220)
        st.caption("Use this when YOLO confuses a cap/hat with a helmet. Save only the selected crop with the correct label.")
        st.write(f"Save path: `{target_dir}`")
        st.write(f"Current `{label}` samples: `{current_count}`")
        if st.button("Save selected crop", key=f"save_headwear_{stable_source}"):
            target_dir = HEADWEAR_SAMPLE_DIR / label
            target_dir.mkdir(parents=True, exist_ok=True)
            preview_crop.save(target_dir / f"{uuid4().hex}.jpg", quality=95)
            new_count = len(list(target_dir.glob("*.jpg")))
            st.success(f"Saved 1 crop. Current `{label}` samples: {new_count}")


def render_result(
    result: DetectionResult,
    source_name: str,
    source_type: str,
    voice_alert: bool,
) -> None:
    details = risk_details(result)
    alert_modes = ["visual", "text"] + (["voice"] if voice_alert and result.alert_count else [])
    message = build_alert_message(result.alert_count)
    summary = "; ".join(f"{item['risk_id']} {item['position']} {item['class']}" for item in details)

    if result.alert_count > 0:
        logger.write_event(
            source_name=source_name,
            source_type=source_type,
            alert_count=result.alert_count,
            total_detections=len(result.detections),
            message=message,
            risk_summary=summary,
            risk_details=details,
            alert_modes=alert_modes,
        )
        speak_alert(message, voice_alert)

    image_col, result_col = st.columns([1.35, 0.85])
    with image_col:
        st.subheader("탐지 결과")
        st.image(result.annotated_image, use_container_width=True)

    with result_col:
        st.subheader("실시간 경고")
        if result.alert_count:
            st.error(message)
        else:
            st.success("현재 입력에서는 안전모 미착용 위험이 감지되지 않았습니다.")

        st.metric("미착용 의심", result.alert_count)
        st.metric("전체 탐지 객체", len(result.detections))

        if details:
            st.subheader("위험 이벤트 상세")
            st.dataframe(pd.DataFrame(details), use_container_width=True)
        else:
            st.write("위험 이벤트 상세가 없습니다.")

        st.subheader("PPE 판정 정책")
        st.dataframe(ppe_policy_summary(result), use_container_width=True)

        if result.detections:
            st.subheader("전체 탐지 목록")
            st.dataframe(pd.DataFrame(result.detections), use_container_width=True)

        render_headwear_sample_collector(result, source_name, source_type)


detector = load_detector()
logger = AlertLogger(LOG_PATH)

st.title("작업자 안전모 착용 자동 탐지 시스템")
st.caption("YOLOv8 안전모 탐지 + 사람 탐지 보조 + 자세 추정 + 멀티모달 경고")

status = detector.status_message()
if detector.is_ready:
    st.success(status)
else:
    st.warning(status)

left, right = st.columns([1.1, 0.9])

with left:
    confidence = st.slider(
        "안전모 탐지 신뢰도 기준",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )
    draw_pose = st.checkbox("사람 자세선 표시", value=False)
    voice_alert = st.checkbox("음성 경고 사용", value=True)

with right:
    st.subheader("탐지 기준")
    st.write(
        "- `helmet`: 안전모 착용\n"
        "- `no_helmet`: 안전모 미착용 머리 영역\n"
        "- `no_helmet_person`: 사람은 보이지만 안전모가 없는 경우\n"
        "- 일반 캡/모자/후드는 안전모로 인정하지 않고 미착용 위험으로 처리\n"
        "- 향후 확장: `helmet`, `cap_hat`, `bare_head` 3분류 PPE 판별 모델\n"
        "- 자세선: YOLOv8 Pose 기반 사람 관절 표시"
    )
    st.info("경고는 시각 박스, 텍스트 메시지, 선택적 음성 안내를 함께 사용합니다.")

image_tab, video_tab, webcam_tab = st.tabs(["이미지", "동영상", "웹캠 스냅샷"])

with image_tab:
    uploaded_image = st.file_uploader(
        "건설 현장 이미지 업로드",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is None:
        st.write("이미지를 업로드하면 안전모 착용 여부, 위험 위치, 판정 사유를 확인할 수 있습니다.")
    else:
        image = Image.open(uploaded_image).convert("RGB")
        with st.spinner("이미지를 분석하는 중입니다..."):
            result = detector.detect(image, confidence=confidence, draw_pose=draw_pose)
        render_result(result, uploaded_image.name, "image", voice_alert)

with video_tab:
    uploaded_video = st.file_uploader(
        "건설 현장 동영상 업로드",
        type=["mp4", "mov", "avi"],
        key="video_upload",
    )
    frame_stride = st.slider("동영상 처리 프레임 간격", 1, 10, 3)
    max_frames = st.slider("최대 처리 프레임 수", 30, 600, 180, step=30)

    if uploaded_video is None:
        st.write("동영상을 업로드하면 프레임별 안전모 미착용과 자세선을 표시한 결과 미리보기를 생성합니다.")
    elif st.button("동영상 분석 시작"):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        input_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
        output_path = VIDEO_DIR / f"{input_path.stem}_annotated.mp4"
        input_path.write_bytes(uploaded_video.getvalue())

        progress = st.progress(0)
        with st.spinner("동영상을 분석하는 중입니다. CPU에서는 시간이 걸릴 수 있습니다..."):
            video_result = process_video(
                input_path=input_path,
                output_path=output_path,
                detector=detector,
                confidence=confidence,
                draw_pose=draw_pose,
                frame_stride=frame_stride,
                max_frames=max_frames,
                progress_callback=progress.progress,
            )

        st.success(
            f"동영상 분석 완료: {video_result.processed_frames}프레임 처리, "
            f"누적 프레임 경고 {video_result.total_alerts}건"
        )
        st.write(f"결과 영상 저장 위치: `{video_result.output_path}`")
        st.write(f"미리보기 저장 위치: `{video_result.preview_path}`")

        if video_result.total_alerts:
            video_risk_details = [
                {
                    **event,
                    "count": 1,
                    "position": "동영상 전체",
                    "reason": reason_for_detection(str(event["class"])),
                    "recommended_action": "위험 구간을 확인하고 안전모 착용 여부를 점검",
                }
                for event in video_result.risk_events
            ]
            video_summary = "; ".join(
                (
                    f"{item['risk_id']} {item['class']} "
                    f"{item['start_time']}s~{item['end_time']}s {item['severity']}"
                )
                for item in video_risk_details
            )
            if video_risk_details:
                st.subheader("동영상 위험 이벤트 요약")
                st.dataframe(pd.DataFrame(video_risk_details), use_container_width=True)
            logger.write_event(
                source_name=uploaded_video.name,
                source_type="video",
                alert_count=video_result.risk_event_count,
                total_detections=video_result.processed_frames,
                message=(
                    f"동영상에서 위험 이벤트 {video_result.risk_event_count}건이 감지되었습니다. "
                    f"프레임 단위 누적 감지는 {video_result.total_alerts}건입니다."
                ),
                risk_summary=video_summary,
                risk_details=video_risk_details,
                alert_modes=["visual", "text"] + (["voice"] if voice_alert else []),
            )

        if False and video_result.total_alerts:
            video_risk_details = [
                {
                    "risk_id": f"V{index:02d}",
                    "class": risk_class,
                    "count": count,
                    "position": "동영상 전체",
                    "reason": reason_for_detection(risk_class),
                    "recommended_action": "위험 프레임을 확인하고 안전모 착용 여부를 점검",
                }
                for index, (risk_class, count) in enumerate(
                    sorted(video_result.risk_type_counts.items()),
                    start=1,
                )
            ]
            video_summary = "; ".join(
                f"{item['risk_id']} {item['class']} {item['count']}건"
                for item in video_risk_details
            )
            logger.write_event(
                source_name=uploaded_video.name,
                source_type="video",
                alert_count=video_result.total_alerts,
                total_detections=video_result.processed_frames,
                message=(
                    f"동영상에서 위험 감지가 누적 {video_result.total_alerts}건 "
                    f"발생했습니다. 위험 프레임 {video_result.risk_frame_count}개를 확인하세요."
                ),
                risk_summary=video_summary,
                risk_details=video_risk_details,
                alert_modes=["visual", "text"] + (["voice"] if voice_alert else []),
            )

        if video_result.preview_path.exists():
            st.subheader("분석 결과 미리보기")
            preview_col, _ = st.columns([0.42, 0.58])
            with preview_col:
                render_gif_preview(video_result.preview_path, width=480)

        if video_result.output_path.exists():
            st.download_button(
                "분석된 영상 다운로드",
                data=video_result.output_path.read_bytes(),
                file_name=video_result.output_path.name,
                mime="video/mp4",
            )

        if video_result.preview_path.exists():
            st.download_button(
                "GIF 미리보기 다운로드",
                data=video_result.preview_path.read_bytes(),
                file_name=video_result.preview_path.name,
                mime="image/gif",
            )

with webcam_tab:
    camera_image = st.camera_input("웹캠으로 현장 이미지를 촬영")
    if camera_image is None:
        st.write("현재는 웹캠 스냅샷 분석을 지원합니다. 실시간 스트리밍은 다음 단계에서 추가할 수 있습니다.")
    else:
        image = ImageOps.mirror(Image.open(camera_image).convert("RGB"))

        with st.spinner("웹캠 이미지를 분석하는 중입니다..."):
            result = detector.detect(image, confidence=confidence, draw_pose=draw_pose)
        render_result(result, "webcam_snapshot", "webcam", voice_alert)

events = logger.read_events()
if events.empty:
    st.divider()
    st.subheader("누적 위험 대시보드")
    st.write("아직 저장된 위험 로그가 없습니다.")
else:
    st.divider()
    render_alert_dashboard(events)
st.stop()

st.divider()
st.subheader("경고 로그")
events = logger.read_events()
if events.empty:
    st.write("아직 저장된 경고 로그가 없습니다.")
else:
    display_columns = [
        column
        for column in [
            "timestamp",
            "source",
            "source_type",
            "alert_count",
            "risk_summary",
            "alert_modes",
            "message",
        ]
        if column in events.columns
    ]
    st.dataframe(events[display_columns], use_container_width=True)

    with st.expander("상세 JSON 로그 보기"):
        detail_column = "risk_details_json"
        if detail_column in events.columns:
            st.dataframe(events[["timestamp", "source", detail_column]], use_container_width=True)
