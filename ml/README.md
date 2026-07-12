# `ml/` — Model, Inference, Evaluation

Everything in this directory is **independent of the web application**. Nothing
here imports FastAPI or `backend/app`, so these modules can be used directly from
notebooks, scripts, or a batch job.

```
ml/
├── model/
│   ├── ffa_net.py     # FFA class — the full network
│   └── blocks.py      # default_conv, PALayer, CALayer, Block, Group
├── inference/
│   └── predictor.py   # standalone CLI: dehaze one image
├── evaluation/
│   └── metrics.py     # PSNR, SSIM, colorfulness, simplified NIQE
└── weights/
    └── its_train_ffa_3_19.pkl   # NOT in git — see below
```

## Weights

The checkpoint is gitignored (~21 MB). Download `its_train_ffa_3_19.pkl` from
<https://github.com/zhilin007/FFA-Net> and place it at
`ml/weights/its_train_ffa_3_19.pkl`.

Full details, including the two loading gotchas (weights nested under
`ckpt['model']`, keys prefixed with `module.`), are in
[`docs/model_info.md`](../docs/model_info.md).

---

## `inference/predictor.py` — CLI

Dehaze a single image from the terminal. Demonstrates that FFA-Net works with no
web stack involved.

```bash
# From the project root, using the backend venv (which has torch + opencv):
backend/.venv/Scripts/python.exe ml/inference/predictor.py \
    --input backend/tests/fixtures/sample_hazy_night.jpg
```

### Arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--input` | *(required)* | Path to the hazy nighttime image. |
| `--output` | `dehazed_<input_name>.jpg` | Where to write the result. |
| `--weights` | `ml/weights/its_train_ffa_3_19.pkl` | Checkpoint path (resolved relative to this package, so it works from any CWD). |
| `--device` | auto (`cuda` if available, else `cpu`) | Compute device. |

### What it does

1. Loads FFA-Net (`gps=3`, `blocks=19`) and the checkpoint, stripping the
   `module.` prefix.
2. Normalises the image to `float32 [0,1]` BGR.
3. Resizes to 512×512, runs inference, restores the original resolution.
4. Applies a light postprocess: CLAHE on the LAB *L* channel + unsharp mask.
5. Saves the result and prints the timing.

### Example output

```
Device : cpu
Weights: .../ml/weights/its_train_ffa_3_19.pkl
Input  : backend/tests/fixtures/sample_hazy_night.jpg  (640x480)
Output : .../dehazed_sample_hazy_night.jpg
Time   : 10.93s
```

> The CLI is intentionally simpler than the API pipeline: it runs **FFA-Net plus
> a basic postprocess only**. It does *not* include glow detection, per-region
> atmospheric light, or the physics-based radiance recovery. For the full
> six-stage pipeline, use the backend
> ([`docs/architecture.md`](../docs/architecture.md)).

---

## `evaluation/metrics.py`

Standalone metric functions, importable without the app:

```python
import numpy as np
from ml.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_colorfulness,
    compute_niqe_simplified,
)

hazy    = cv2.imread("hazy.jpg")     # uint8 BGR
dehazed = cv2.imread("dehazed.jpg")  # uint8 BGR

print(compute_psnr(hazy, dehazed))          # higher is better
print(compute_ssim(hazy, dehazed))          # higher is better, 0–1
print(compute_colorfulness(dehazed))        # Hasler & Susstrunk
print(compute_niqe_simplified(dehazed))     # lower is more natural
```

All functions take **uint8 BGR** images (OpenCV convention).

> `compute_ssim` uses scikit-image's `channel_axis` parameter. The older
> `multichannel=True` keyword was removed in scikit-image 0.19+ and will raise.

---

## `model/`

`ffa_net.py` and `blocks.py` reproduce the official FFA-Net architecture
**unmodified** — layer names and structure must match the released checkpoint or
`load_state_dict(..., strict=True)` fails. Do not refactor these files.

Verify the model loads:

```bash
backend/.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, '.')
import torch
from ml.model.ffa_net import FFA
m = FFA(gps=3, blocks=19)
ckpt = torch.load('ml/weights/its_train_ffa_3_19.pkl', map_location='cpu')
sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
m.load_state_dict(sd, strict=True)
print('OK —', sum(p.numel() for p in m.parameters()), 'parameters')
"
# OK — 4455913 parameters
```
