from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_DIR = PROJECT_ROOT

RISK_CLASSES = {"head", "no_helmet", "no-hardhat", "no hardhat", "without_helmet", "cap", "hat", "cap_hat"}
SAFE_CLASSES = {"helmet", "hardhat", "hard_hat"}
REFERENCE_CLASSES = {"person", "worker"}
HEADWEAR_MODEL_PATH = PROJECT_ROOT / "models" / "headwear_cls.pt"
HEADWEAR_SAFE_CLASSES = {"helmet", "hardhat", "hard_hat", "safety_helmet"}
HEADWEAR_RISK_CLASSES = {"cap", "hat", "cap_hat", "bare_head", "head", "no_helmet"}
HEADWEAR_STRONG_CAP_THRESHOLD = 0.90
HEADWEAR_WEAK_HELMET_SCORE = 0.65
HEADWEAR_WEAK_HELMET_CAP_THRESHOLD = 0.86
HEADWEAR_RISK_BOX_CAP_THRESHOLD = 0.86
HEADWEAR_BARE_HEAD_THRESHOLD = 0.90
HEADWEAR_WEAK_HELMET_BARE_HEAD_THRESHOLD = 0.60
HEADWEAR_HELMET_RESCUE_THRESHOLD = 0.88
HEADWEAR_SAFE_LOCK_SCORE = 0.98
HEADWEAR_CAP_MARGIN_OVER_BARE = 0.04
HEADWEAR_CAP_MARGIN_OVER_SAFE = 0.06
HEADWEAR_BARE_MARGIN_OVER_CAP = 0.14
HEADWEAR_BARE_MARGIN_OVER_SAFE = 0.12
VISUAL_CAP_PURPLE_RATIO = 0.025
VISUAL_CAP_PINK_RATIO = 0.025
DISPLAY_LABELS = {
    "head": "no_helmet",
    "helmet": "helmet",
    "cap": "cap_hat",
    "hat": "cap_hat",
    "cap_hat": "cap_hat",
    "bare_head": "no_helmet",
    "person": "person",
}


@dataclass
class DetectionResult:
    annotated_image: Image.Image
    detections: list[dict[str, Any]]
    alert_count: int
    source_image: Image.Image | None = None


class SafetyHelmetDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.person_model = None
        self.pose_model = None
        self.headwear_model = None
        self.names: dict[int, str] = {}
        self.headwear_names: dict[int, str] = {}
        self.load_error: str | None = None
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.load_error = f"ultralytics 패키지가 아직 설치되지 않았습니다: {exc}"
            return

        try:
            self.model = YOLO(str(self.model_path)) if self.model_path.exists() else YOLO("yolov8n.pt")
            self.person_model = YOLO("yolov8n.pt")
            self.pose_model = YOLO("yolov8n-pose.pt")
            if HEADWEAR_MODEL_PATH.exists():
                self.headwear_model = YOLO(str(HEADWEAR_MODEL_PATH))
                self.headwear_names = dict(self.headwear_model.names)
            self.names = dict(self.model.names)
        except Exception as exc:
            self.load_error = f"YOLO 모델을 불러오지 못했습니다: {exc}"

    def status_message(self) -> str:
        if self.model is None:
            return self.load_error or "모델이 아직 준비되지 않았습니다."
        if self.model_path.exists():
            return f"학습 모델을 사용 중입니다: {self.model_path} + 사람/자세 탐지 보조 모델"
        return "학습 모델이 없어 기본 YOLOv8n 모델을 사용 중입니다. 안전모 전용 학습 후 models/best.pt를 넣으세요."

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        draw_pose: bool = False,
        risk_confidence: float | None = None,
        person_fallback_confidence: float = 0.55,
        min_person_area_ratio: float = 0.010,
        show_people: bool = False,
        imgsz: int = 640,
    ) -> DetectionResult:
        if self.model is None:
            return DetectionResult(
                annotated_image=image,
                detections=[],
                alert_count=0,
                source_image=image,
            )

        image_array = np.array(image)
        risk_confidence = confidence if risk_confidence is None else risk_confidence
        predictions = self.model.predict(image_array, conf=confidence, imgsz=imgsz, verbose=False)
        raw_detections = self._helmet_detections(predictions[0], image_array, risk_confidence)
        raw_detections.extend(
            self._detect_people_without_helmets(
                image_array,
                raw_detections,
                person_fallback_confidence=person_fallback_confidence,
                min_person_area_ratio=min_person_area_ratio,
            )
        )
        if show_people:
            raw_detections.extend(
                self._person_reference_detections(
                    image_array,
                    confidence=max(0.35, person_fallback_confidence),
                    min_person_area_ratio=min_person_area_ratio,
                )
            )

        detections = self._suppress_safe_overlaps(raw_detections)
        alert_count = sum(1 for detection in detections if detection["status"] == "risk")

        annotated = image_array.copy()
        if draw_pose:
            annotated = self._draw_pose(annotated)

        for detection in detections:
            self._draw_detection(annotated, detection)

        return DetectionResult(
            annotated_image=Image.fromarray(annotated),
            detections=detections,
            alert_count=alert_count,
            source_image=image,
        )

    def _helmet_detections(
        self,
        result: Any,
        image_array: np.ndarray,
        risk_confidence: float,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = self.names.get(class_id, str(class_id))
            normalized_label = label.lower().strip()
            status = self._status_for_label(normalized_label)
            if status == "risk" and score < risk_confidence:
                continue

            image_height = image_array.shape[0]
            if status == "risk" and y1 <= 2 and y2 <= image_height * 0.12:
                continue

            display_label = DISPLAY_LABELS.get(normalized_label, label)
            verification = None

            if status in {"safe", "risk"}:
                verification = self._verify_headwear(
                    image_array,
                    x1,
                    y1,
                    x2,
                    y2,
                    prefer_safe=status == "risk",
                )
                if verification:
                    if status == "safe" and self._should_override_helmet(score, verification):
                        status = "risk"
                        display_label = DISPLAY_LABELS.get(verification["class"], verification["class"])
                    elif status == "risk" and self._should_rescue_helmet(verification):
                        status = "safe"
                        display_label = "helmet"
                    elif status == "risk" and verification["status"] == "risk" and verification["class"] == "cap_hat":
                        display_label = "cap_hat"

                if status == "risk" and display_label != "cap_hat" and normalized_label in RISK_CLASSES:
                    risk_box_cap = self._verify_headwear(
                        image_array,
                        x1,
                        y1,
                        x2,
                        y2,
                        prefer_cap_for_risk=True,
                    )
                    if risk_box_cap and risk_box_cap["class"] == "cap_hat":
                        verification = risk_box_cap
                        display_label = "cap_hat"

            if status == "risk" and y1 <= 2 and y2 <= image_array.shape[0] * 0.12:
                continue

            detection = {
                "class": display_label,
                "confidence": round(score, 3),
                "status": status,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            if verification:
                detection["headwear_check"] = verification["class"]
                detection["headwear_confidence"] = verification["confidence"]
            detections.append(detection)
        return detections

    def _verify_headwear(
        self,
        image_array: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        prefer_cap_for_risk: bool = False,
        prefer_safe: bool = False,
    ) -> dict[str, Any] | None:
        crops = self._headwear_crop_candidates(image_array, x1, y1, x2, y2)
        if not crops:
            return None

        for crop in crops:
            visual_hint = self._visual_headwear_hint(crop, prefer_cap=prefer_cap_for_risk)
            if visual_hint == "cap_hat":
                return {"class": "cap_hat", "confidence": 0.98, "status": "risk", "source": "visual"}

        if self.headwear_model is None:
            return None

        best_cap: dict[str, Any] | None = None
        best_no_helmet: dict[str, Any] | None = None
        best_safe: dict[str, Any] | None = None

        for crop in crops:
            result = self.headwear_model.predict(crop, imgsz=224, verbose=False)[0]
            if result.probs is None:
                continue

            scores = result.probs.data.detach().cpu().tolist()
            for class_id, score in enumerate(scores):
                label = self.headwear_names.get(class_id, str(class_id)).lower().strip()
                display_label = DISPLAY_LABELS.get(label, label)
                candidate = {"class": display_label, "confidence": round(float(score), 3), "source": "model"}

                if display_label == "cap_hat":
                    if best_cap is None or score > float(best_cap["confidence"]):
                        best_cap = candidate
                elif label in HEADWEAR_SAFE_CLASSES:
                    if best_safe is None or score > float(best_safe["confidence"]):
                        best_safe = candidate
                elif label in HEADWEAR_RISK_CLASSES:
                    if best_no_helmet is None or score > float(best_no_helmet["confidence"]):
                        best_no_helmet = candidate

        cap_threshold = 0.55 if prefer_cap_for_risk else HEADWEAR_STRONG_CAP_THRESHOLD
        cap_score = float(best_cap["confidence"]) if best_cap else 0.0
        no_helmet_score = float(best_no_helmet["confidence"]) if best_no_helmet else 0.0
        safe_score = float(best_safe["confidence"]) if best_safe else 0.0
        score_summary = {
            "cap_hat": round(cap_score, 3),
            "no_helmet": round(no_helmet_score, 3),
            "helmet": round(safe_score, 3),
        }

        if prefer_safe and best_safe and safe_score >= HEADWEAR_HELMET_RESCUE_THRESHOLD and safe_score >= cap_score - 0.08:
            return {**best_safe, "status": "safe", "scores": score_summary}

        cap_is_clear = (
            best_cap
            and (
                cap_score >= cap_threshold
                or (not prefer_cap_for_risk and cap_score >= 0.97 and safe_score < 0.88)
            )
            and cap_score >= no_helmet_score - (0.05 if prefer_cap_for_risk else -HEADWEAR_CAP_MARGIN_OVER_BARE)
            and (cap_score >= safe_score + HEADWEAR_CAP_MARGIN_OVER_SAFE or (prefer_cap_for_risk and safe_score < 0.86) or cap_score >= 0.97)
        )
        if cap_is_clear:
            return {**best_cap, "status": "risk", "scores": score_summary}
        if best_safe and safe_score >= HEADWEAR_HELMET_RESCUE_THRESHOLD and safe_score >= cap_score - 0.05:
            return {**best_safe, "status": "safe", "scores": score_summary}
        bare_head_is_clear = (
            best_no_helmet
            and no_helmet_score >= HEADWEAR_BARE_HEAD_THRESHOLD
            and no_helmet_score >= cap_score + HEADWEAR_BARE_MARGIN_OVER_CAP
            and no_helmet_score >= safe_score + HEADWEAR_BARE_MARGIN_OVER_SAFE
        )
        if bare_head_is_clear:
            return {**best_no_helmet, "status": "risk", "scores": score_summary}
        if best_cap:
            return {**best_cap, "status": "unknown", "scores": score_summary}
        if best_safe:
            return {**best_safe, "status": "unknown", "scores": score_summary}
        if best_no_helmet:
            return {**best_no_helmet, "status": "unknown", "scores": score_summary}
        return None

    def _headwear_crop_candidates(
        self,
        image_array: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> list[np.ndarray]:
        height, width = image_array.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 1 or box_height <= 1:
            return []

        regions = []
        regions.append((x1, y1, x2, y2, 0.12, 0.18))

        top_y2 = y1 + int(box_height * 0.52)
        regions.append((x1, y1, x2, top_y2, 0.12, 0.15))

        top_height = max(1, int(box_height * 0.48))
        top_y1 = y1
        top_y2 = y1 + top_height
        slice_width = max(1, int(box_width * 0.62))
        starts = [
            x1,
            x1 + int(box_width * 0.19),
            max(x1, x2 - slice_width),
        ]
        for start_x in starts:
            regions.append((start_x, top_y1, start_x + slice_width, top_y2, 0.10, 0.12))

        crops: list[np.ndarray] = []
        seen: set[tuple[int, int, int, int]] = set()
        for rx1, ry1, rx2, ry2, pad_x_ratio, pad_y_ratio in regions:
            region_width = max(1, rx2 - rx1)
            region_height = max(1, ry2 - ry1)
            pad_x = int(region_width * pad_x_ratio)
            pad_y = int(region_height * pad_y_ratio)
            crop_x1 = max(0, rx1 - pad_x)
            crop_y1 = max(0, ry1 - pad_y)
            crop_x2 = min(width, rx2 + pad_x)
            crop_y2 = min(height, ry2 + pad_y)
            key = (crop_x1, crop_y1, crop_x2, crop_y2)
            if key in seen or crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue
            seen.add(key)
            crop = image_array[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size:
                crops.append(crop)

        return crops

    def _visual_headwear_hint(self, crop: np.ndarray, prefer_cap: bool = False) -> str | None:
        if crop.size == 0:
            return None
        # Color-only rules caused hard hats in compressed video frames to flip into cap/hat.
        # Keep this hook disabled and let the trained second-stage classifier decide.
        return None

    def _should_override_helmet(self, helmet_score: float, verification: dict[str, Any]) -> bool:
        if verification["status"] != "risk":
            return False

        headwear_class = verification["class"]
        headwear_score = float(verification["confidence"])

        if headwear_class == "cap_hat":
            if verification.get("source") == "visual":
                return True
            if helmet_score >= HEADWEAR_SAFE_LOCK_SCORE:
                return False
            if headwear_score >= 0.97 and helmet_score < HEADWEAR_SAFE_LOCK_SCORE:
                return True
            if helmet_score >= 0.93 and verification.get("source") != "visual":
                return False
            threshold = (
                HEADWEAR_WEAK_HELMET_CAP_THRESHOLD
                if helmet_score < HEADWEAR_WEAK_HELMET_SCORE
                else HEADWEAR_STRONG_CAP_THRESHOLD
            )
            return headwear_score >= threshold

        if headwear_class == "no_helmet":
            if helmet_score >= HEADWEAR_SAFE_LOCK_SCORE:
                return False
            threshold = (
                HEADWEAR_WEAK_HELMET_BARE_HEAD_THRESHOLD
                if helmet_score < HEADWEAR_WEAK_HELMET_SCORE
                else HEADWEAR_BARE_HEAD_THRESHOLD
            )
            return headwear_score >= threshold

        return headwear_score >= HEADWEAR_BARE_HEAD_THRESHOLD

    def _should_rescue_helmet(self, verification: dict[str, Any]) -> bool:
        if verification["status"] != "safe":
            return False
        return float(verification["confidence"]) >= HEADWEAR_HELMET_RESCUE_THRESHOLD

    def _draw_detection(self, image_array: np.ndarray, detection: dict[str, Any]) -> None:
        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]
        color = self._color_for_status(detection["status"])

        cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 3)
        caption_confidence = detection.get("headwear_confidence", detection["confidence"])
        caption = f"{detection['class']} {caption_confidence:.2f}"
        cv2.putText(
            image_array,
            caption,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_pose(self, image_array: np.ndarray) -> np.ndarray:
        if self.pose_model is None:
            return image_array

        pose_result = self.pose_model.predict(image_array, conf=0.25, verbose=False)[0]
        if pose_result.keypoints is None:
            return image_array

        skeleton = [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]
        colors = [
            (0, 255, 255),
            (255, 0, 255),
            (255, 0, 255),
            (0, 180, 255),
            (0, 180, 255),
            (0, 255, 120),
            (0, 255, 120),
            (255, 120, 0),
            (255, 120, 0),
            (255, 120, 0),
            (80, 160, 255),
            (80, 160, 255),
        ]

        points = pose_result.keypoints.xy.cpu().numpy()
        confidences = pose_result.keypoints.conf
        confidence_values = confidences.cpu().numpy() if confidences is not None else None

        for person_index, person_points in enumerate(points):
            person_conf = confidence_values[person_index] if confidence_values is not None else None
            for line_index, (joint_a, joint_b) in enumerate(skeleton):
                if person_conf is not None and (person_conf[joint_a] < 0.30 or person_conf[joint_b] < 0.30):
                    continue
                x1, y1 = person_points[joint_a]
                x2, y2 = person_points[joint_b]
                if min(x1, y1, x2, y2) <= 0:
                    continue
                cv2.line(
                    image_array,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    colors[line_index],
                    3,
                    cv2.LINE_AA,
                )

            for joint_index, (x, y) in enumerate(person_points):
                if person_conf is not None and person_conf[joint_index] < 0.30:
                    continue
                if x <= 0 or y <= 0:
                    continue
                cv2.circle(image_array, (int(x), int(y)), 4, (0, 220, 255), -1, cv2.LINE_AA)

        return image_array

    def _status_for_label(self, label: str) -> str:
        if label in RISK_CLASSES:
            return "risk"
        if label in SAFE_CLASSES:
            return "safe"
        if label in REFERENCE_CLASSES:
            return "reference"
        return "unknown"

    def _suppress_safe_overlaps(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        safe_boxes = [detection for detection in detections if detection["status"] == "safe"]
        filtered: list[dict[str, Any]] = []

        for detection in detections:
            if detection["status"] != "risk":
                filtered.append(detection)
                continue

            overlaps_safe_helmet = any(
                self._intersection_over_area(detection, safe_box) > 0.25
                for safe_box in safe_boxes
            )
            if not overlaps_safe_helmet:
                filtered.append(detection)

        return filtered

    def _detect_people_without_helmets(
        self,
        image_array: np.ndarray,
        helmet_detections: list[dict[str, Any]],
        person_fallback_confidence: float,
        min_person_area_ratio: float,
    ) -> list[dict[str, Any]]:
        if self.person_model is None:
            return []

        predictions = self.person_model.predict(
            image_array,
            conf=person_fallback_confidence,
            classes=[0],
            verbose=False,
        )
        safe_boxes = [detection for detection in helmet_detections if detection["status"] == "safe"]
        risk_boxes = [detection for detection in helmet_detections if detection["status"] == "risk"]
        fallback_detections: list[dict[str, Any]] = []

        image_height, image_width = image_array.shape[:2]
        min_person_area = image_width * image_height * min_person_area_ratio

        for box in predictions[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0].item())
            if (x2 - x1) * (y2 - y1) < min_person_area:
                continue

            upper_region = self._person_upper_region(x1, y1, x2, y2)
            has_helmet = any(self._box_center_inside(safe_box, upper_region) for safe_box in safe_boxes)
            already_has_head_alert = any(
                self._intersection_over_area(risk_box, upper_region) > 0.20 for risk_box in risk_boxes
            )

            verification = self._verify_headwear(
                image_array,
                upper_region["x1"],
                upper_region["y1"],
                upper_region["x2"],
                upper_region["y2"],
                prefer_cap_for_risk=True,
            )

            if already_has_head_alert:
                continue

            if upper_region["y1"] <= 2 and upper_region["y2"] <= image_array.shape[0] * 0.12:
                continue

            if has_helmet:
                continue

            if verification and verification["status"] == "safe":
                continue

            fallback_class = "no_helmet"
            if (
                verification
                and verification["status"] == "risk"
                and verification["class"] == "cap_hat"
                and float(verification["confidence"]) >= HEADWEAR_RISK_BOX_CAP_THRESHOLD
            ):
                fallback_class = "cap_hat"

            detection = {
                "class": fallback_class,
                "confidence": round(score, 3),
                "status": "risk",
                "x1": upper_region["x1"],
                "y1": upper_region["y1"],
                "x2": upper_region["x2"],
                "y2": upper_region["y2"],
            }
            if verification:
                detection["headwear_check"] = verification["class"]
                detection["headwear_confidence"] = verification["confidence"]
            fallback_detections.append(detection)

        return fallback_detections

    def _person_reference_detections(
        self,
        image_array: np.ndarray,
        confidence: float,
        min_person_area_ratio: float,
    ) -> list[dict[str, Any]]:
        if self.person_model is None:
            return []

        predictions = self.person_model.predict(
            image_array,
            conf=confidence,
            classes=[0],
            verbose=False,
        )
        image_height, image_width = image_array.shape[:2]
        min_person_area = image_width * image_height * min_person_area_ratio
        detections: list[dict[str, Any]] = []

        for box in predictions[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0].item())
            if (x2 - x1) * (y2 - y1) < min_person_area:
                continue
            detections.append(
                {
                    "class": "person",
                    "confidence": round(score, 3),
                    "status": "reference",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return detections

    def _person_upper_region(self, x1: int, y1: int, x2: int, y2: int) -> dict[str, Any]:
        height = y2 - y1
        width = x2 - x1
        return {
            "x1": max(0, x1 + int(width * 0.10)),
            "y1": max(0, y1),
            "x2": max(0, x2 - int(width * 0.10)),
            "y2": max(0, y1 + int(height * 0.38)),
        }

    def _box_center_inside(self, box: dict[str, Any], region: dict[str, Any]) -> bool:
        x_center = (box["x1"] + box["x2"]) / 2
        y_center = (box["y1"] + box["y2"]) / 2
        return region["x1"] <= x_center <= region["x2"] and region["y1"] <= y_center <= region["y2"]

    def _intersection_over_area(self, first: dict[str, Any], second: dict[str, Any]) -> float:
        x_left = max(first["x1"], second["x1"])
        y_top = max(first["y1"], second["y1"])
        x_right = min(first["x2"], second["x2"])
        y_bottom = min(first["y2"], second["y2"])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        first_area = max(1, (first["x2"] - first["x1"]) * (first["y2"] - first["y1"]))
        return intersection / first_area

    def _color_for_status(self, status: str) -> tuple[int, int, int]:
        if status == "risk":
            return (255, 0, 0)
        if status == "safe":
            return (0, 170, 0)
        if status == "reference":
            return (255, 170, 0)
        return (120, 120, 120)
