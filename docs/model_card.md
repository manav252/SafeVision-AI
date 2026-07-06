# SafeVision AI Model Card

## Model Purpose

SafeVision AI uses YOLO/OpenCV-based computer vision to support an industrial safety demo. The model path is intended to detect workers and PPE-related conditions in CCTV-style footage, then combine detection metadata with plant context such as gas readings and restricted-zone status.

## Model Type

- Object detection model: YOLOv8-compatible `.pt` file.
- Demo fallback: YOLOv8n person detection with OpenCV color heuristics for helmet/vest estimation when a custom PPE model is unavailable.

## Expected Inputs

- Image or video frames from CCTV-style industrial footage.
- Optional metadata: camera ID, zone name, PPE status, gas readings, restricted-zone breach state, and confidence score.

## Expected Outputs

- Detected object class, such as `person`, `helmet`, `no helmet`, `vest`, or related PPE labels when supported by the custom model.
- Bounding box metadata.
- Confidence score between 0 and 1.
- Derived PPE status and risk factors used by SafeVision AI.

## Confidence Score Meaning

The confidence score represents the detector's confidence in a detected object/class for a frame. It is not a guarantee of safety compliance and should be interpreted with the surrounding context, camera quality, and model limitations.

## Dataset Assumptions

The current repository does not include a documented training dataset, train/validation split, annotation schema, or benchmark report. The included model assets should therefore be treated as demo assets unless separate training documentation is supplied.

## Evaluation Metrics

No verified real-world evaluation metrics are included in this repository. SafeVision AI does not fabricate precision, recall, mAP, or false alarm rates.

Recommended evaluation workflow:

1. Collect representative plant CCTV footage with appropriate consent and safety approvals.
2. Annotate workers, PPE, restricted zones, and relevant safety conditions.
3. Split data into train, validation, and holdout test sets.
4. Report precision, recall, F1, mAP@50, mAP@50:95, false-positive rate, and false-negative rate.
5. Evaluate separately by lighting, camera angle, PPE color, occlusion, and zone type.
6. Document model version, dataset version, thresholds, and known failure cases.

## Limitations

- Demo footage may not represent real plant conditions.
- PPE color heuristics can fail under poor lighting, motion blur, unusual PPE colors, reflective surfaces, or occlusion.
- Person detection without tracking can count the same worker across multiple frames.
- Detection confidence is not the same as operational safety certainty.
- The model should not be used as the sole decision-maker for real safety enforcement.

## Safety Limitations

SafeVision AI is a decision-support demo. Production deployment would require site-specific validation, human review, incident-response procedures, sensor calibration, privacy review, and integration with approved safety systems.

## Ethical Considerations

- Avoid using worker identity features without consent, governance, and legal review.
- Minimize retention of identifiable footage.
- Provide human appeal/review workflows for safety warnings.
- Monitor bias across PPE styles, body types, lighting conditions, camera positions, and work environments.

## Future Improvements

- Add model versioning and a model registry.
- Publish dataset documentation and training pipeline details.
- Add object tracking to reduce duplicate alerts.
- Add real evaluation metrics after a verified holdout evaluation.
- Add model cards per model version.
