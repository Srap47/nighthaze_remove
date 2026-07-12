# Model Information — FFA-Net

## Overview

NightHaze uses **FFA-Net** (Feature Fusion Attention Network) as the deep
learning core of its dehazing pipeline. It is an end-to-end, fully convolutional
network that maps a hazy image directly to a clean one, with no explicit
estimation of transmission or atmospheric light.

| Property | Value |
| -------- | ----- |
| Architecture | FFA-Net (Feature Fusion Attention Network) |
| Paper | AAAI 2020 |
| Checkpoint | `its_train_ffa_3_19.pkl` |
| Groups (`gps`) | 3 |
| Blocks per group (`blocks`) | 19 |
| Feature dimension | 64 |
| Parameters | **4,455,913** (~4.46M) |
| Training data | RESIDE — Indoor Training Set (ITS) |
| Inference input | 512×512 (resized, then restored to original size) |
| CPU inference | ~10–11 s |

## Why attention?

FFA-Net's central insight is that **haze is not uniform** — neither across
colour channels nor across space. Two attention mechanisms encode this:

- **Channel Attention (CA)** — haze affects the R, G and B channels unevenly
  (scattering is wavelength-dependent). CA learns per-channel weights via global
  average pooling → 1×1 conv → ReLU → 1×1 conv → sigmoid.
- **Pixel Attention (PA)** — haze density varies spatially; thin haze near the
  camera, dense haze far away. PA learns a per-pixel weight map, letting the
  network spend capacity on the hazier regions.

### Structure

```
Input (3ch)
   │
   ▼  pre-conv
Features (64ch)
   │
   ├─▶ Group 1 ──┐   each Group = 19 × Block + conv, with a long skip
   ├─▶ Group 2 ──┤   each Block  = conv → ReLU → conv → CA → PA → residual
   └─▶ Group 3 ──┤
                 ▼
        Feature Fusion (concat 3×64ch → channel attention → weighted sum)
                 │
                 ▼  Pixel Attention → post-conv
        Output (3ch) + global residual with input
```

The **feature fusion** step is what gives the network its name: rather than
using only the last group's output, it concatenates all three groups and learns
adaptive weights over them, preserving shallow detail alongside deep semantics.
The final global residual (`out + input`) means the network learns the *haze
residual* rather than reconstructing the image from scratch.

---

## Weights

### Download

The checkpoint is **not committed to this repository** (see `.gitignore` —
`ml/weights/*.pkl`). Obtain it from the official implementation:

- **Repository:** <https://github.com/zhilin007/FFA-Net>
- **File:** `its_train_ffa_3_19.pkl` (~21 MB), from the repo's
  `trained_models/` directory.

### Placement

```
ml/weights/its_train_ffa_3_19.pkl
```

This matches `MODEL_WEIGHTS_PATH` in `.env.example`
(`../ml/weights/its_train_ffa_3_19.pkl`, resolved relative to `backend/`).

If the file is absent, the API **still starts** — it logs a warning, reports
`model_loaded: false` from `/api/v1/health`, and returns `503 model_not_loaded`
from the dehazing endpoints.

### Checkpoint format — two gotchas

The `.pkl` is a **full training snapshot**, not a bare `state_dict`:

```python
ckpt.keys()
# ['step', 'max_psnr', 'max_ssim', 'ssims', 'psnrs', 'losses', 'model']
```

1. The weights live under **`ckpt['model']`**, not at the top level.
2. Every key is prefixed with **`module.`** because the model was trained under
   `nn.DataParallel`. This prefix must be stripped or `load_state_dict` fails
   with a full set of missing/unexpected keys.

```python
import torch
from ml.model.ffa_net import FFA

model = FFA(gps=3, blocks=19)
ckpt = torch.load("ml/weights/its_train_ffa_3_19.pkl", map_location="cpu")

state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(state_dict, strict=True)   # strict=True passes cleanly
model.eval()
```

Loading with `strict=True` succeeds with zero missing or unexpected keys, which
confirms the local architecture matches the official one exactly.

---

## Domain gap: indoor training, nighttime use

The checkpoint is trained on **RESIDE ITS** — synthetically hazed *indoor*,
*daylight* scenes. NightHaze applies it to outdoor night photography, so the
domain gap is real and deliberate. It is acceptable because:

- The network learns an **illumination-agnostic appearance prior** — where haze
  concentrates in space and across channels. That structure transfers.
- The properties that *don't* transfer — multiple coloured light sources,
  non-uniform atmospheric light, glow halos — are exactly what the pipeline's
  **physics stage** handles explicitly (see
  [architecture.md](architecture.md#4-design-decisions)).

Training a nighttime-specific model would require paired (hazy, clean) night
images of identical scenes, which do not exist at usable scale.

---

## Citation

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

The architecture in [`ml/model/`](../ml/model/) is reproduced from the authors'
official implementation without modification, so that the released weights load
exactly.
