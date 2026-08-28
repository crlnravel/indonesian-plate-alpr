---
language:
- id
- en
license: mit
tags:
- yolo
- yolo11
- ultralytics
- object-detection
- ocr
- alpr
- anpr
- character-recognition
- indonesia
pipeline_tag: object-detection
---

# Indonesian License Plate Character Recognizer (YOLO11)

A fine-tuned **YOLO11 Nano** character recognition model trained to detect and classify 36 alphanumeric characters (`0-9`, `A-Z`) on cropped Indonesian vehicle license plates.

## Model Performance
- **Precision**: 96.24%
- **Recall**: 96.34%
- **mAP@0.5**: 98.40%
- **mAP@0.5:0.95**: 72.01%
- **Inference Latency**: 2.41 ms (~415 FPS)

## Model Summary
- **Task**: 36-Class Alphanumeric Character Detection
- **Classes**: `0-9`, `A-Z` (36 classes)
- **Base Model**: `yolo11n.pt` (Ultralytics)
- **Dataset**: [Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)
- **Export Formats**: PyTorch (`.pt`), ONNX (`.onnx`), TFLite INT8 (`.tflite`)

## Usage with Ultralytics
```python
from ultralytics import YOLO

model = YOLO("plate_recognizer_yolo11n.pt")
results = model.predict("cropped_plate.jpg", conf=0.35)
results[0].show()
```
