from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from src.detector import SafetyHelmetDetector


@dataclass
class VideoProcessResult:
    output_path: Path
    preview_path: Path
    processed_frames: int
    total_alerts: int
    risk_frame_count: int
    risk_type_counts: dict[str, int]
    risk_event_count: int
    risk_events: list[dict[str, float | int | str]]
    fps: float


def _severity_for_event(duration_seconds: float, detection_count: int, peak_count: int) -> str:
    if duration_seconds >= 3.0 or peak_count >= 2:
        return "Critical"
    if duration_seconds >= 1.0 or detection_count >= 3:
        return "High"
    if detection_count >= 2:
        return "Medium"
    return "Low"


def _close_event(event: dict[str, float | int | str]) -> dict[str, float | int | str]:
    duration = max(0.0, float(event["end_time"]) - float(event["start_time"]))
    detection_count = int(event["detection_count"])
    peak_count = int(event["peak_count"])
    return {
        "risk_id": "",
        "class": str(event["class"]),
        "start_time": round(float(event["start_time"]), 2),
        "end_time": round(float(event["end_time"]), 2),
        "duration_seconds": round(duration, 2),
        "frame_count": int(event["frame_count"]),
        "detection_count": detection_count,
        "peak_count": peak_count,
        "severity": _severity_for_event(duration, detection_count, peak_count),
    }


def _update_risk_events(
    active_events: dict[str, dict[str, float | int | str]],
    completed_events: list[dict[str, float | int | str]],
    frame_risk_counts: dict[str, int],
    timestamp_seconds: float,
    merge_gap_seconds: float,
) -> None:
    for risk_class, event in list(active_events.items()):
        gap_seconds = timestamp_seconds - float(event["end_time"])
        if risk_class in frame_risk_counts and gap_seconds <= merge_gap_seconds:
            continue
        if gap_seconds > merge_gap_seconds:
            completed_events.append(_close_event(event))
            del active_events[risk_class]

    for risk_class, count in frame_risk_counts.items():
        event = active_events.get(risk_class)
        if event is None:
            active_events[risk_class] = {
                "class": risk_class,
                "start_time": timestamp_seconds,
                "end_time": timestamp_seconds,
                "frame_count": 1,
                "detection_count": count,
                "peak_count": count,
            }
            continue

        event["end_time"] = timestamp_seconds
        event["frame_count"] = int(event["frame_count"]) + 1
        event["detection_count"] = int(event["detection_count"]) + count
        event["peak_count"] = max(int(event["peak_count"]), count)


def _finalize_risk_events(
    active_events: dict[str, dict[str, float | int | str]],
    completed_events: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    events = completed_events + [_close_event(event) for event in active_events.values()]
    events.sort(key=lambda event: (float(event["start_time"]), str(event["class"])))
    for index, event in enumerate(events, start=1):
        event["risk_id"] = f"V{index:02d}"
    return events


def process_video(
    input_path: Path,
    output_path: Path,
    detector: SafetyHelmetDetector,
    confidence: float,
    draw_pose: bool,
    frame_stride: int = 3,
    max_frames: int = 300,
    progress_callback: Callable[[float], None] | None = None,
) -> VideoProcessResult:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    expected_frames = min((total_frames + frame_stride - 1) // frame_stride, max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / max(1, frame_stride)),
        (width, height),
    )

    frame_index = 0
    processed_frames = 0
    total_alerts = 0
    risk_frame_count = 0
    risk_type_counts: dict[str, int] = {}
    active_events: dict[str, dict[str, float | int | str]] = {}
    completed_events: list[dict[str, float | int | str]] = []
    event_merge_gap_seconds = 2.0
    preview_frames: list[Image.Image] = []
    max_preview_frames = 80

    while frame_index < total_frames and processed_frames < max_frames:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if frame_index % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(
                Image.fromarray(frame_rgb),
                confidence=confidence,
                draw_pose=draw_pose,
                risk_confidence=max(confidence, 0.50),
                person_fallback_confidence=max(confidence, 0.50),
                min_person_area_ratio=0.012,
                imgsz=960,
            )
            annotated_bgr = cv2.cvtColor(np.array(result.annotated_image), cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

            if len(preview_frames) < max_preview_frames and processed_frames % 2 == 0:
                preview_image = result.annotated_image.copy()
                preview_image.thumbnail((720, 720))
                preview_frames.append(preview_image)

            processed_frames += 1
            total_alerts += result.alert_count
            if result.alert_count:
                risk_frame_count += 1
                frame_risk_counts: dict[str, int] = {}
                for detection in result.detections:
                    if detection["status"] == "risk":
                        risk_class = detection["class"]
                        risk_type_counts[risk_class] = risk_type_counts.get(risk_class, 0) + 1
                        frame_risk_counts[risk_class] = frame_risk_counts.get(risk_class, 0) + 1
                _update_risk_events(
                    active_events,
                    completed_events,
                    frame_risk_counts,
                    frame_index / max(1.0, fps),
                    event_merge_gap_seconds,
                )
            elif active_events:
                _update_risk_events(
                    active_events,
                    completed_events,
                    {},
                    frame_index / max(1.0, fps),
                    event_merge_gap_seconds,
                )

            if progress_callback is not None:
                progress_callback(min(1.0, processed_frames / max(1, expected_frames)))

        frame_index += 1

    capture.release()
    writer.release()

    preview_path = output_path.with_suffix(".gif")
    if preview_frames:
        preview_frames[0].save(
            preview_path,
            save_all=True,
            append_images=preview_frames[1:],
            duration=max(80, int(1000 / max(1.0, fps / max(1, frame_stride)))),
            loop=0,
        )

    risk_events = _finalize_risk_events(active_events, completed_events)

    return VideoProcessResult(
        output_path=output_path,
        preview_path=preview_path,
        processed_frames=processed_frames,
        total_alerts=total_alerts,
        risk_frame_count=risk_frame_count,
        risk_type_counts=risk_type_counts,
        risk_event_count=len(risk_events),
        risk_events=risk_events,
        fps=fps,
    )
