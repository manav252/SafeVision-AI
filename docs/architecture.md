# SafeVision AI Architecture

## Data Flow

```text
Uploaded video
    |
    v
Streamlit app.py
    |
    |-- first frame -> drawable canvas -> restricted-zone polygon
    |-- gas readings + permit context -> compound-risk context
    |-- fire event -> biometric override context
    |
    v
Background processing thread
    |
    v
OpenCV frame reader
    |
    |-- every 3rd frame
    |-- resize to 640px width inside detector
    v
detector.py
    |
    |-- models/ppe_yolov8.pt available -> custom PPE detection
    |-- model unavailable -> yolov8n.pt person detection + HSV PPE heuristics
    v
risk_engine.py
    |
    |-- restricted zone breach: +40
    |-- no helmet: +30
    |-- no vest: +25
    |-- hazardous gas accumulation: +35
    |-- gas accumulation + active permit: +45
    |-- gas accumulation + equipment condition: +20
    |-- gas accumulation + shift changeover: +15
    |-- audit deviation + active risk: +20
    |-- fire biometric override: +50
    |-- cap score at 100
    |-- generate administrator hazard messages and response states
    v
utils.py
    |
    |-- draw bounding boxes and zone overlay
    |-- save evidence frames
    |-- export administrator alert CSV logs
    v
Streamlit UI
    |
    |-- live detection frame
    |-- geospatial plant heatmap
    |-- automated response workflow
    |-- compound risk co-pilot reasoning chain
    |-- what-if permit simulation
    |-- incident report draft
    |-- administrator alert queue
```

## Component Responsibilities

| Component | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI, upload handling, canvas zone drawing, worker thread orchestration, live progress and log display. |
| `detector.py` | YOLOv8 loading, Apple MPS/CPU device selection, inference, custom PPE parsing, fallback HSV PPE estimation. |
| `risk_engine.py` | Per-frame risk scoring, gas accumulation scoring, named worker warnings, compound permit-risk detection, fire biometric override alerts, violation row generation, severity and risk-level classification. |
| `utils.py` | Folder creation, default zone generation, canvas polygon parsing, point-in-polygon checks, drawing annotations, evidence and CSV output. |
| `models/` | Optional custom PPE model location. Place `ppe_yolov8.pt` here. |
| `outputs/evidence/` | Saved violation screenshots. |
| `outputs/logs/` | Exported violation CSV files. |
| `sample_videos/` | Optional local video clips for repeatable reviews. |

## PS1 Intelligence Layer

SafeVision AI is designed around the core idea in Problem Statement 1: a plant should not only detect isolated events, it should correlate weak signals into compound risk.

| Signal | Data Source | Intelligence Output |
| --- | --- | --- |
| CCTV person detection | YOLOv8 / fallback YOLOv8n | Worker presence, PPE state, restricted-zone breach |
| Gas accumulation | Simulated SCADA feed | CH4, CO, H2S, and O2 hazard status |
| Permit-to-work | Sidebar scenario controls | Maintenance, hot work, or confined-space overlap |
| Equipment maintenance | Sidebar condition controls | Maintenance, bypass, and isolation overlap risk |
| Shift changeover | Sidebar handover controls | Supervisor acknowledgement and handover-risk signal |
| Worker identity | Simulated registry | Named administrator warning messages |
| Fire event | Sidebar emergency selector | Biometric access override workflow |
| Plant layout | Streamlit heatmap | Zone A/B/C risk status for operational review |
| Regulatory context | Built-in guidance rules | Corrective action recommendations and incident report draft |
| What-if permit | User-selected permit candidate | Block/review/allow recommendation before approval |
| Knowledge graph | Built-in relationship graph | Equipment-permit-risk relationships across plant signals |

## Safety Co-Pilot Agents

SafeVision AI presents a deterministic multi-agent reasoning chain so the interface remains reliable without depending on an external LLM API.

| Agent | Input | Output |
| --- | --- | --- |
| Vision Agent | YOLO detections, PPE estimates, zone polygon | Worker/PPE/restricted-zone finding |
| Gas Sensor Agent | CH4, CO, H2S, O2 telemetry | Normal, gas accumulation, compound risk, or emergency override state |
| Permit Agent | Active permit and maintenance flag | Whether active work overlaps with hazardous conditions |
| Equipment Agent | Maintenance and isolation status | Equipment-risk overlap with hazardous conditions |
| Shift Agent | Shift handover and crew status | Handover-risk warning during abnormal conditions |
| Historical Pattern Agent | Near-miss pattern memory | Pattern match for gas + active work + equipment/shift context |
| Compliance Audit Agent | OISD, DGMS, Factory Act checklist status | Corrective action workflow recommendation |
| Risk Orchestrator | All agent outputs plus risk score | Pause/warn/monitor recommendation |

In production, this layer can be replaced with LangGraph, CrewAI, or another agent framework connected to governed plant data and regulatory RAG sources.

## Detection Modes

### Custom PPE Mode

When `models/ppe_yolov8.pt` exists, SafeVision AI uses it directly. The code expects the model to detect at least some of these labels:

- `person`
- `helmet`, `hardhat`, or `hard hat`
- `no helmet`, `no hardhat`, or `no hard hat`
- `safety vest`, `vest`, `hi-vis vest`, or `high visibility vest`
- `no vest` or `no safety vest`

### Fallback Mode

When the custom PPE model is unavailable, SafeVision AI uses `yolov8n.pt` for `person` detection and estimates PPE:

- Head region: top 30% of the person bounding box.
- Helmet signal: bright yellow or white HSV cluster.
- Torso region: middle 40% of the person bounding box.
- Vest signal: high-visibility orange or green HSV cluster.
- Missing PPE rows are marked as `estimated`.

## Known Limitations

- HSV heuristics are sensitive to lighting, camera exposure, motion blur, and PPE color variation.
- The fallback mode can confuse bright backgrounds with helmets or vests.
- Gas readings are currently supplied by the app controls and can be connected to real SCADA or gas detector hardware.
- Permit context is manually selected in the UI rather than integrated with a permit-to-work system.
- Worker identity should be connected to approved badge, access-control, or consent-based biometric systems.
- Biometric access control override should be connected to physical access-control systems before use on site.
- The geospatial heatmap can be replaced with a real GIS/CAD plant layout.
- Regulatory guidance is rule-based text, not legal advice or a certified compliance engine.
- Custom PPE class names vary by dataset; aliases may need adjustment in `detector.py`.
- Streamlit reruns the script frequently, so long-running processing is intentionally isolated in a background thread.
- Evidence saving can produce many images for long videos with persistent violations.
- The app is optimized for focused review sessions, not high-throughput production CCTV streams.

## Future Work

- Train or fine-tune a dedicated PPE YOLOv8 model for the target industrial environment.
- Connect live gas detector, SCADA historian, and permit-to-work APIs.
- Integrate administrator notification channels such as SMS, email, Teams, or control-room alarms.
- Integrate approved access-control systems for emergency biometric override.
- Add time-series gas trend forecasting to predict accumulation before thresholds are crossed.
- Replace the built-in heatmap with a real plant GIS/CAD layout and live zone telemetry.
- Add a regulatory RAG agent that drafts incident reports against site SOPs, Factory Act references, and emergency response checklists.
- Add object tracking to avoid repeatedly counting the same person across adjacent frames.
- Add configurable risk weights and restricted-zone presets.
- Add support for multiple zones with different severity levels.
- Add video export with annotations.
- Add role-based review workflows for safety officers.
- Add cloud object storage for evidence retention.
