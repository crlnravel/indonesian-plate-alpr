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
- alpr
- anpr
- license-plate-detection
- indonesia
pipeline_tag: object-detection
---

# Indonesian Vehicle License Plate Detector (YOLO11)

A fine-tuned **YOLO11 Nano** object detection model trained to detect Indonesian vehicle license plates (cars, motorcycles, commercial vehicles) across varied weather and lighting conditions.

## Model Performance
- **Precision**: 94.13%
- **Recall**: 95.53%
- **mAP@0.5**: 98.84%
- **mAP@0.5:0.95**: 70.86%
- **Inference Latency**: 6.91 ms (~145 FPS)

## Model Summary
- **Task**: Single-class License Plate Detection (`0: license_plate`)
- **Base Model**: `yolo11n.pt` (Ultralytics)
- **Dataset**: [Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)
- **Export Formats**: PyTorch (`.pt`), ONNX (`.onnx`), TFLite INT8 (`.tflite`)

## Usage with Ultralytics
```python
from ultralytics import YOLO

model = YOLO("plate_detector_yolo11n.pt")
results = model.predict("vehicle.jpg", conf=0.35)
results[0].show()
```

## License
This model and its weights are open-source and released under the **MIT License**.
