# PPE Model Training Guide

SafeVision AI looks for a trained PPE model at:

```text
models/ppe_yolov8.pt
```

If that file exists, the app uses it for PPE detection. If it does not exist, the app uses YOLOv8n person detection plus fallback PPE estimation.

## 1. Classes

Train these classes first:

```text
0: person
1: helmet
2: safety_vest
```

Avoid training `no_helmet` and `no_vest` initially. Those are absence states, not strong visual objects. SafeVision can infer missing PPE by checking whether helmet and vest detections overlap the person head/torso regions.

## 2. Dataset Folder

Export a PPE dataset in YOLO format and place it like this:

```text
SafeVision-AI/
├── datasets/
│   └── ppe/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
└── training/
    └── ppe_data.yaml
```

Every image should have a matching `.txt` file in the corresponding `labels/` folder.

## 3. Labeling Rules

Draw boxes around:

- Full visible worker body as `person`
- Hard hat as `helmet`
- High-visibility jacket/vest as `safety_vest`

Include difficult examples:

- Worker without helmet
- Worker without vest
- Side views
- Far CCTV views
- Low light
- Partial body
- Different helmet colors
- Workers close to machinery

For a hackathon model, target at least 300-500 labeled images. For a stronger model, target 1,000-3,000 images.

## 4. Train On Mac M1

Training on Mac is possible but slower:

```bash
cd SafeVision-AI
source .venv/bin/activate
python training/train_ppe_model.py --device mps --epochs 50 --batch 4
```

If MPS gives memory issues, use CPU:

```bash
python training/train_ppe_model.py --device cpu --epochs 50 --batch 4
```

## 5. Train On Google Colab

Colab is recommended for faster training.

Upload the project or just the dataset and run:

```bash
pip install ultralytics
yolo detect train model=yolov8n.pt data=ppe_data.yaml epochs=50 imgsz=640 batch=8
```

After training, download:

```text
runs/detect/train/weights/best.pt
```

Rename it:

```text
ppe_yolov8.pt
```

Place it into:

```text
SafeVision-AI/models/ppe_yolov8.pt
```

## 6. Validate

```bash
python training/validate_ppe_model.py
```

Important metrics:

- Recall: how many real PPE items the model finds
- Precision: how many detections are correct
- mAP50: general detection quality
- False negatives: missed helmet/vest detections

For safety, recall matters a lot because missed violations are more dangerous than extra warnings.

## 7. Test On A Video

```bash
python training/predict_ppe_sample.py sample_videos/factory_demonstration.mp4
```

Predictions are saved under:

```text
runs/ppe_predictions/sample/
```

## 8. Use In SafeVision AI

Once `models/ppe_yolov8.pt` exists:

```bash
python -m streamlit run app.py
```

The detector will load the trained PPE model automatically.
