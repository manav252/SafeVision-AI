# Sample JSON Payloads

## PPE Violation Event

```json
{
  "zone_name": "Zone B",
  "event_type": "no_vest",
  "message": "Safety vest not detected for worker near pump maintenance area",
  "worker_id": "WKR-118",
  "evidence_uri": "outputs/evidence/zone_b_no_vest.jpg",
  "metadata_json": {
    "camera": "CCTV-2",
    "bbox": [210, 140, 480, 720],
    "confidence": 0.82
  }
}
```

## Restricted-Zone Entry Event

```json
{
  "zone_name": "Reactor Zone",
  "event_type": "restricted_zone_entry",
  "message": "Worker entered restricted reactor zone during elevated gas condition",
  "worker_id": "WKR-221",
  "metadata_json": {
    "permit": "Maintenance Permit",
    "gas": {
      "ch4_lel": 14,
      "co_ppm": 46,
      "h2s_ppm": 8,
      "oxygen_percent": 20.1
    }
  }
}
```

## Camera Configuration

```json
{
  "name": "CCTV-3",
  "stream_url": "rtsp://factory-camera.local/reactor-zone",
  "zone_name": "Reactor Zone",
  "restricted_zone": {
    "type": "polygon",
    "points": [[80, 320], [460, 300], [520, 620], [110, 650]]
  }
}
```

