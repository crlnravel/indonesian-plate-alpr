#!/usr/bin/env python3
"""
Indonesian License Plate Recognition (ALPR) - Gradio Web App
Deployable directly on Hugging Face Spaces.
"""

import os
import cv2
import gradio as gr
import numpy as np
from pipeline import IndonesianALPR

DETECTOR_PATH = os.environ.get("DETECTOR_PATH", "plate_detector_yolo11n.pt")
RECOGNIZER_PATH = os.environ.get("RECOGNIZER_PATH", "plate_recognizer_yolo11n.pt")

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = IndonesianALPR(
            detector_weights=DETECTOR_PATH,
            recognizer_weights=RECOGNIZER_PATH,
            det_conf=0.35,
            rec_conf=0.35
        )
    return pipeline

def recognize_plate(image, det_thresh, rec_thresh):
    if image is None:
        return None, "Please upload an image."

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    alpr = get_pipeline()
    alpr.det_conf = det_thresh
    alpr.rec_conf = rec_thresh

    results = alpr.predict(img_bgr)
    annotated_bgr = alpr.annotate_image(img_bgr, results)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if not results:
        return annotated_rgb, "No license plate detected. Try lowering the detection threshold."

    summary = []
    for i, r in enumerate(results, 1):
        summary.append(
            f"**Plate #{i}**: `{r['recognized_text'] or '[Unreadable]'}` "
            f"(Confidence: {r['detector_confidence'] * 100:.1f}%, Characters: {r['char_count']})"
        )

    return annotated_rgb, "\n".join(summary)

with gr.Blocks(title="🇮🇩 Indonesian License Plate Recognition (ALPR)") as demo:
    gr.Markdown("""
    # 🇮🇩 Indonesian Automatic License Plate Recognition (ALPR)
    Real-time, two-stage computer vision pipeline powered by **YOLO11**:
    1. **Stage 1 (Detection)**: Localizes vehicle license plate bounding boxes.
    2. **Stage 2 (OCR / Recognition)**: Detects and transcribes alphanumeric characters (`0-9`, `A-Z`).
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Vehicle Image")
            with gr.Row():
                det_slider = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Plate Detection Threshold")
                rec_slider = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Character Recognition Threshold")
            submit_btn = gr.Button("🔍 Detect & Recognize License Plate", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Annotated Result")
            output_text = gr.Markdown(label="Recognition Output")

    submit_btn.click(
        fn=recognize_plate,
        inputs=[input_image, det_slider, rec_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()
