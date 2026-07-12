# API Reference

Base URL (development): `http://localhost:8000`
All application routes are versioned under `/api/v1`.

Interactive docs are served at `/docs` (Swagger UI) and `/redoc`.

---

## Conventions

### Images

Every image in a response is a **base64-encoded PNG data URI**, directly usable
as an `<img src>`:

```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
```

### Errors

**All** failures — domain errors, HTTP errors, validation errors, and unexpected
crashes — return the same envelope:

```json
{
  "success": false,
  "error": "not_found",
  "detail": "Not Found"
}
```

| `error` | HTTP | Cause |
| ------- | ---- | ----- |
| `bad_request` | 400 | Unsupported upload content type |
| `invalid_image` | 400 | File is not a decodable image, or is too small |
| `not_found` | 404 | Unknown route, or the demo fixture is missing |
| `payload_too_large` | 413 | Upload exceeds `MAX_IMAGE_SIZE_MB` |
| `image_too_large` | 413 | Image rejected by size limits |
| `validation_error` | 422 | Malformed request (e.g. missing `image` field) |
| `pipeline_error` | 500 | A pipeline stage failed (message names the stage) |
| `internal_error` | 500 | Unexpected error (details are logged, never returned) |
| `model_not_loaded` | 503 | FFA-Net weights missing or failed to load |
| `http_error` | other | Any other HTTP error (e.g. 405) |

---

## `GET /`

Welcome message.

**Response `200`**

```json
{
  "message": "NightHaze API",
  "docs": "/docs"
}
```

---

## `GET /api/v1/health`

Service health and model readiness. Use this to confirm the weights loaded — the
service starts successfully **even when weights are missing**, reporting
`model_loaded: false` rather than crashing.

**Response `200`**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "FFA-Net (its_train_ffa_3_19)",
  "device": "cpu",
  "version": "1.0.0"
}
```

---

## `POST /api/v1/dehaze/upload`

Dehaze an uploaded image.

**Request** — `multipart/form-data`

| Field | Type | Notes |
| ----- | ---- | ----- |
| `image` | file | **Required.** `image/jpeg`, `image/png`, or `image/webp`. Max 10 MB. |

```bash
curl -X POST http://localhost:8000/api/v1/dehaze/upload \
  -F "image=@night_photo.jpg;type=image/jpeg"
```

Images larger than 2048px on the long side are **downscaled**, not rejected.
Processing is CPU-bound and takes roughly **12–15 seconds** (FFA-Net inference
dominates), so set a generous client timeout — the bundled frontend uses 180s.

**Response `200`** — `DehazeResponse` (abbreviated; base64 payloads truncated)

```json
{
  "success": true,
  "job_id": "3f2b9c1a-5d7e-4a91-b0c4-8e6f1d2a7b3c",
  "original_image_b64":   "data:image/png;base64,iVBORw0KGgo...",
  "dehazed_image_b64":    "data:image/png;base64,iVBORw0KGgo...",
  "transmission_map_b64": "data:image/png;base64,iVBORw0KGgo...",
  "glow_mask_b64":        "data:image/png;base64,iVBORw0KGgo...",
  "metrics": {
    "psnr": 17.952,
    "ssim": 0.8063,
    "niqe": 1.107,
    "brisque": 71.715,
    "visibility_score": 0.3155,
    "colorfulness_before": 10.109,
    "colorfulness_after": 16.914,
    "colorfulness_improvement_pct": 67.32,
    "processing_time_ms": 12076.91
  },
  "pipeline_stages": [
    { "stage": "preprocessing",      "time_ms": 29.9 },
    { "stage": "glow_detection",     "time_ms": 10.0 },
    { "stage": "ffa_net_inference",  "time_ms": 10679.1 },
    { "stage": "radiance_recovery",  "time_ms": 62.6 },
    { "stage": "postprocessing",     "time_ms": 913.5 },
    { "stage": "quality_assessment", "time_ms": 381.8 }
  ]
}
```

### Response fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `job_id` | string | UUID for this run. |
| `original_image_b64` | string | The input **after** the processing-cap resize, so it matches the output dimensions exactly. |
| `dehazed_image_b64` | string | Final result. |
| `transmission_map_b64` | string | Grayscale transmission visualisation. |
| `glow_mask_b64` | string | Detected light sources / halos, green overlay. |
| `metrics` | object | See below. |
| `pipeline_stages` | array | Per-stage wall time, always 6 entries in pipeline order. |

### Metrics

| Metric | Direction | Meaning |
| ------ | --------- | ------- |
| `psnr` | higher better | Peak Signal-to-Noise Ratio (dB), input vs output. |
| `ssim` | higher better | Structural Similarity, 0–1. |
| `niqe` | lower better | Simplified naturalness score. |
| `brisque` | lower better | No-reference perceptual quality, 0–100. **`-1` means unavailable** — treat it as "no score", not as a good score. |
| `visibility_score` | higher better | Estimated fraction of haze removed. |
| `colorfulness_before` / `_after` | — | Hasler & Susstrunk colorfulness. |
| `colorfulness_improvement_pct` | — | Percentage change; can be negative. |
| `processing_time_ms` | — | Sum of all stage times. |

**Error responses**

```jsonc
// 400 — wrong content type
{ "success": false, "error": "bad_request",
  "detail": "Unsupported content type 'text/plain'. Allowed: image/jpeg, image/png, image/webp." }

// 413 — file over the size limit
{ "success": false, "error": "payload_too_large",
  "detail": "Image exceeds the 10 MB size limit." }

// 422 — no file supplied
{ "success": false, "error": "validation_error",
  "detail": "body.image: Field required" }

// 503 — weights not loaded
{ "success": false, "error": "model_not_loaded",
  "detail": "FFA-Net model is not loaded. Download 'its_train_ffa_3_19.pkl' from ..." }
```

---

## `GET /api/v1/dehaze/demo`

Runs the bundled sample nighttime image
(`backend/tests/fixtures/sample_hazy_night.jpg`) through the full pipeline. Lets
the frontend demonstrate the system without an upload.

```bash
curl http://localhost:8000/api/v1/dehaze/demo
```

**Response `200`** — identical schema to `/dehaze/upload`.

**Response `404`** — if the fixture is absent:

```json
{ "success": false, "error": "not_found",
  "detail": "Demo fixture not found at .../sample_hazy_night.jpg." }
```

---

## Configuration

Server behaviour is controlled by environment variables (see `.env.example`):

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `DEBUG` | `false` | Enables DEBUG-level logging. |
| `MODEL_WEIGHTS_PATH` | `../ml/weights/its_train_ffa_3_19.pkl` | Resolved relative to `backend/`. |
| `DEVICE` | `cpu` | `cpu` or `cuda`. |
| `MAX_IMAGE_SIZE_MB` | `10` | Upload size limit. |

CORS defaults to allowing `http://localhost:5173` and `http://localhost:3000`.
