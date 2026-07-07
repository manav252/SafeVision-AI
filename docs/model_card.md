# SafeVision AI PPE Detection Model Card

## Overview

SafeVision AI uses a YOLOv8-based object detection model for person and PPE detection as part of a broader industrial safety platform. The model supports CCTV-style video analysis in the Streamlit dashboard and helps identify safety-relevant visual signals such as people and PPE-related conditions.

Detection outputs are combined with contextual plant information by the SafeVision AI risk engine. This context can include gas readings, permits, restricted zones, equipment status, shift handover notes, compliance checklist state, and emergency status. The combined signal is used to generate risk scores, safety events, alerts, dashboard summaries, and AI Safety Advisor recommendations.

## Model Information

- **Architecture:** YOLOv8
- **Framework:** Ultralytics
- **Source:** Roboflow-exported pretrained PPE detection model
- **Fine-tuning:** None in this repository
- **Inference:** Real-time through Streamlit/OpenCV integration

The model was not trained from scratch for this project. The repository uses a pretrained, Roboflow-exported PPE detection model as a demo/prototype component.

## Intended Use

This model is intended for:

- academic demonstrations
- software engineering projects
- industrial safety research
- prototype safety monitoring

This model is not certified for production industrial safety, regulatory compliance, or autonomous safety enforcement.

## System Role

```text
Camera Feed
    |
    v
YOLOv8 Detection
    |
    v
Risk Engine
    |
    v
AI Safety Advisor
    |
    v
Alerts
    |
    v
Dashboard
    |
    v
PostgreSQL
```

The model provides visual detection signals. SafeVision AI then combines those signals with operational context and stores resulting safety events and alerts through the FastAPI backend.

## Current Capabilities

- Person detection in CCTV-style frames
- PPE detection when supported by the loaded pretrained model classes
- Restricted-zone monitoring through application-level zone geometry
- Risk-score generation through the SafeVision AI risk engine
- AI Safety Advisor integration for contextual recommendations
- Event logging through Streamlit, FastAPI, and PostgreSQL

## Current Limitations

- Uses a pretrained Roboflow-exported model.
- No project-specific fine-tuning has been performed.
- No formal benchmark has been conducted on an independent industrial dataset.
- Performance depends on camera quality, lighting, viewing angle, motion blur, occlusion, PPE appearance, and scene layout.
- Detection confidence should not be interpreted as a guarantee of compliance or safety.
- The current implementation is intended for demonstrations and research only.

## Evaluation Metrics

Formal evaluation has not yet been conducted for this repository. No accuracy, precision, recall, F1-score, mAP, dataset size, training epochs, or benchmark numbers are claimed.

A future evaluation should use a documented, independent industrial PPE dataset and report metrics such as Precision, Recall, F1-score, mAP@50, and mAP@50:95. Evaluation should also break down performance by lighting condition, camera angle, PPE class, occlusion level, and site environment.

## Ethical Considerations

- AI predictions should support, not replace, trained human supervisors.
- False positives and false negatives are possible.
- Human verification is recommended before operational decisions or incident escalation.
- Any deployment involving worker footage should address privacy, consent, retention, access control, and local legal requirements.
- The system should be validated for fairness across PPE styles, body types, camera positions, lighting conditions, and work environments before real-world use.

## Future Improvements

- Fine-tune on industrial PPE datasets.
- Collect representative real plant data with appropriate consent and governance.
- Benchmark using Precision, Recall, F1-score, and mAP.
- Support multiple PPE classes with clear class definitions.
- Improve robustness in low-light and high-motion conditions.
- Explore edge deployment for site-local inference.
- Add continuous model monitoring for drift, false positives, and false negatives.
