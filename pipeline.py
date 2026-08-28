#!/usr/bin/env python3
"""
Indonesian Automatic License Plate Recognition (ALPR) Pipeline
Stage 1: Vehicle License Plate Detection (YOLO11)
Stage 2: Plate Character Recognition & Text Assembly (YOLO11)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# 36 Class Names for Indonesian License Plate Characters (0-9, A-Z)
CHAR_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


def resolve_weight_path(weight_name_or_path: str | Path) -> str:
    """Helper to locate weight files across current directory, script directory, or weights subfolder."""
    p = Path(weight_name_or_path)
    if p.exists():
        return str(p)
    candidates = [
        Path(__file__).parent / p.name,
        Path(__file__).parent / "weights" / p.name,
        Path.cwd() / p.name,
        Path.cwd() / "weights" / p.name,
        Path.cwd() / "BahanPortfolio" / "indonesian-plate-alpr" / "weights" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(weight_name_or_path)


class IndonesianALPR:
    """
    Two-stage Automatic License Plate Recognition (ALPR) for Indonesian plates.
    """

    def __init__(
        self,
        detector_weights: str | Path = "plate_detector_yolo11n.pt",
        recognizer_weights: str | Path = "plate_recognizer_yolo11n.pt",
        det_conf: float = 0.35,
        rec_conf: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.det_conf = det_conf
        self.rec_conf = rec_conf
        self.device = device

        resolved_det = resolve_weight_path(detector_weights)
        resolved_rec = resolve_weight_path(recognizer_weights)

        print(f"Loading Plate Detector: {resolved_det}")
        self.detector = YOLO(resolved_det)

        print(f"Loading Character Recognizer: {resolved_rec}")
        self.recognizer = YOLO(resolved_rec)

    @staticmethod
    def reconstruct_plate_string(boxes_data: List[dict]) -> str:
        """
        Sort detected character bounding boxes horizontally (X-axis) and format into a readable plate number.
        """
        if not boxes_data:
            return ""

        sorted_chars = sorted(boxes_data, key=lambda item: item['x_center'])
        raw_text = "".join([item['char'] for item in sorted_chars])
        return raw_text

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        return_crops: bool = True
    ) -> List[Dict]:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            img = image_input.copy()

        h, w = img.shape[:2]

        det_results = self.detector.predict(
            source=img,
            conf=self.det_conf,
            device=self.device,
            verbose=False
        )

        detections = []
        if not det_results or len(det_results[0].boxes) == 0:
            return detections

        boxes = det_results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0].item())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = img[y1:y2, x1:x2]

            char_results = self.recognizer.predict(
                source=plate_crop,
                conf=self.rec_conf,
                device=self.device,
                verbose=False
            )

            detected_chars = []
            if char_results and len(char_results[0].boxes) > 0:
                for c_box in char_results[0].boxes:
                    c_xyxy = c_box.xyxy[0].cpu().numpy().astype(int)
                    c_cls = int(c_box.cls[0].item())
                    c_conf = float(c_box.conf[0].item())
                    c_char = CHAR_CLASSES[c_cls] if c_cls < len(CHAR_CLASSES) else "?"

                    detected_chars.append({
                        'char': c_char,
                        'conf': round(c_conf, 3),
                        'x_center': (c_xyxy[0] + c_xyxy[2]) / 2,
                        'y_center': (c_xyxy[1] + c_xyxy[3]) / 2,
                        'bbox_rel': c_xyxy.tolist()
                    })

            plate_text = self.reconstruct_plate_string(detected_chars)

            plate_info = {
                'plate_bbox': [int(x1), int(y1), int(x2), int(y2)],
                'detector_confidence': round(conf, 3),
                'recognized_text': plate_text,
                'char_count': len(detected_chars),
                'characters': detected_chars,
            }

            if return_crops:
                plate_info['plate_crop'] = plate_crop

            detections.append(plate_info)

        return detections

    def annotate_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        detections: Optional[List[Dict]] = None
    ) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        else:
            img = image_input.copy()

        if detections is None:
            detections = self.predict(img, return_crops=False)

        for det in detections:
            x1, y1, x2, y2 = det['plate_bbox']
            text = det['recognized_text'] or "Plate"
            conf = det['detector_confidence']

            label = f"{text} ({conf:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - lbl_h - 8), (x1 + lbl_w + 6, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return img
