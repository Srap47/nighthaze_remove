<div align="center">

# NightHaze

**Nighttime haze removal — a hybrid deep learning + physics pipeline.**

Daytime dehazing assumes one light source and a uniform sky. Night breaks both
assumptions. NightHaze pairs a pretrained FFA-Net backbone with an explicit
atmospheric-scattering model that estimates illumination **per light source**.

Final Year B.Tech CSE Project

[Architecture](docs/architecture.md) ·
[API Reference](docs/api_reference.md) ·
[Model Info](docs/model_info.md) ·
[ML Tools](ml/README.md)

</div>

---

## Why nighttime is different

| | Daytime | Nighttime |
| --- | --- | --- |
| Light sources | One (the sun) | Many — lamps, neon, headlights |
| Atmospheric light `A` | Globally constant | Varies per region |
| Dark Channel Prior | Works | **Fails** — no dark channel near bright lights |
| Glow halos | Rare | Everywhere |
| Sensor noise | Low | High (low light ⇒ high ISO) |

NightHaze addresses each of these directly: it detects individual light sources,
estimates a **per-pixel atmospheric light map**, excludes glow regions from the
dark channel so the prior stays usable, and blends back toward the neural output
where the physics is numerically weakest.

## Features

- **Six-stage pipeline** — preprocessing → glow detection → FFA-Net → atmospheric
  light + radiance recovery → post-processing → quality assessment.
- **Per-region atmospheric light** with radial blending at glow boundaries.
- **Glow-aware recovery** that avoids the over-brightening classical inversion
  produces near light sources.
- **Explainable output** — the API returns the transmission map and the glow
  mask alongside the result, not just the final image.
- **Six real metrics** — PSNR, SSIM, NIQE, BRISQUE, visibility, colorfulness —
  plus per-stage timings.
- **Interactive UI** — drag-to-compare before/after slider, animated metrics.
- **Degrades gracefully** — the API starts and reports `model_loaded: false` if
  weights are missing, instead of crashing.

---

## Quick start

**Prerequisites:** Python 3.11, Node 18+, [uv](https://github.com/astral-sh/uv).

### 1. Model weights

The checkpoint is not in git. Download `its_train_ffa_3_19.pkl` (~21 MB) from
[zhilin007/FFA-Net](https://github.com/zhilin007/FFA-Net) and place it at:

```
ml/weights/its_train_ffa_3_19.pkl
```

### 2. Backend

```bash
cd backend
uv venv --python 3.11
uv pip install -r requirements.txt

cp ../.env.example .env          # optional
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

API on **http://localhost:8000** · Swagger UI at **/docs**.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

UI on **http://localhost:5173**.

### 4. Verify

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","model_loaded":true,"model_name":"FFA-Net (its_train_ffa_3_19)", ...}
```

> **First run is slow.** FFA-Net inference on CPU takes **~10–11 s**, so a full
> request is **~12–15 s** end to end. That is expected, not a hang. Use the
> "Try with Demo Image" button to see it without uploading anything.

### Tests

```bash
cd backend
uv pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests/ -v

# skip the ~15s full-pipeline tests:
.venv/Scripts/python.exe -m pytest tests/ -v -m "not slow"
```

---

## Architecture at a glance

```
uint8 BGR ─▶ 1. Preprocess ─┬─▶ 2. Glow Detect ──┐
                            │                     ├─▶ 4. Atm Light + Recovery
                            └─▶ 3. FFA-Net ───────┘             │
                                                                ▼
                              6. Metrics ◀── 5. Post-process ───┘
                                    │
                                    ▼
                            DehazeResponse (JSON)
```

Internally, **all images are `float32` BGR in `[0,1]`; `uint8` appears only at
I/O boundaries.**

### The physics

Hazy image formation, and its inversion:

```
I(x) = J(x)·t(x) + A·(1 − t(x))          J(x) = (I(x) − A) / max(t(x), t₀) + A
```

Where NightHaze departs from the textbook: `A` is a **per-pixel map**, not a
constant, built by blending each light source's local atmospheric light toward a
glow-excluded global estimate. Full derivation in
[docs/architecture.md](docs/architecture.md).

---

## Screenshots

> _Placeholder — add captures before submission._

| | |
| --- | --- |
| **Landing page** <br> `docs/images/landing.png` | **Before/after slider** <br> `docs/images/comparison.png` |
| **Metrics dashboard** <br> `docs/images/metrics.png` | **Transmission map & glow mask** <br> `docs/images/maps.png` |

---

## Tech stack

| Layer | Technology | Purpose |
| ----- | ---------- | ------- |
| **Model** | PyTorch 2.1 (CPU) | FFA-Net inference |
| | FFA-Net (AAAI 2020) | 4.46M-param dehazing backbone |
| **Backend** | FastAPI 0.110 | Async API, auto OpenAPI docs |
| | Pydantic 2 / pydantic-settings | Schemas, typed config |
| | OpenCV (headless) | Classical CV, image I/O |
| | NumPy · SciPy · scikit-image | Physics, metrics |
| | uvicorn | ASGI server |
| **Frontend** | React 19 + TypeScript | UI |
| | Vite 8 | Build tooling |
| | Tailwind CSS 3 | Styling |
| | axios · react-router · react-dropzone | HTTP, routing, upload |
| **Tooling** | uv | Python env + dependency management |
| | pytest | 31 backend tests |

---

## API summary

Base URL `http://localhost:8000`, routes under `/api/v1`.

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/` | Welcome message. |
| `GET` | `/api/v1/health` | Status + whether the model loaded. |
| `POST` | `/api/v1/dehaze/upload` | Dehaze an uploaded image (`multipart`, field `image`; JPEG/PNG/WebP ≤ 10 MB). |
| `GET` | `/api/v1/dehaze/demo` | Dehaze the bundled sample image. |

Both dehaze endpoints return the same payload: four base64 PNGs (original,
dehazed, transmission map, glow mask), the metrics object, and six stage
timings. **Every** error — including 404s and validation failures — uses one
envelope:

```json
{ "success": false, "error": "payload_too_large", "detail": "Image exceeds the 10 MB size limit." }
```

Full request/response examples: [docs/api_reference.md](docs/api_reference.md).

---

## Standalone CLI

FFA-Net runs without the web stack:

```bash
backend/.venv/Scripts/python.exe ml/inference/predictor.py \
    --input backend/tests/fixtures/sample_hazy_night.jpg
```

See [ml/README.md](ml/README.md).

---

## Project layout

```
nighthaze/
├── backend/          FastAPI app, services, tests
│   └── app/
│       ├── api/          routes + router
│       ├── core/         pipeline, exceptions, logging
│       ├── models/       Pydantic schemas
│       └── services/     the 7 pipeline services
├── ml/               model, standalone CLI, metrics, weights
├── frontend/         React + TypeScript UI
└── docs/             architecture, API reference, model info
```

---

## Citation

This project builds on FFA-Net; the architecture is reproduced unmodified so the
authors' released weights load exactly.

> Qin, X., Wang, Z., Bai, Y., Xie, X., & Jia, H. (2020). FFA-Net: Feature Fusion
> Attention Network for Single Image Dehazing. *Proceedings of the AAAI
> Conference on Artificial Intelligence*, 34(07), 11908–11915.

```bibtex
@inproceedings{qin2020ffa,
  title     = {FFA-Net: Feature Fusion Attention Network for Single Image Dehazing},
  author    = {Qin, Xu and Wang, Zhilin and Bai, Yuanchao and Xie, Xiaodong and Jia, Huizhu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {34},
  number    = {07},
  pages     = {11908--11915},
  year      = {2020}
}
```

Haze model: He, K., Sun, J., & Tang, X. (2011). *Single Image Haze Removal Using
Dark Channel Prior.* IEEE TPAMI, 33(12), 2341–2353.
