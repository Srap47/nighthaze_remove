# Architecture

NightHaze is a six-stage pipeline that combines a pretrained deep learning model
(FFA-Net) with an explicit physical model of atmospheric scattering. The deep
model does the heavy lifting; the physics stage corrects it using per-region
estimates of the scene's illumination.

---

## 1. Pipeline overview

```
                    ┌──────────────────────────────────────┐
   uint8 BGR  ───▶  │  1. Preprocessing                    │
   (upload)         │     validate → ensure BGR → resize   │
                    │     → denoise → gamma lift           │
                    └──────────────────┬───────────────────┘
                                       │  float32 BGR [0,1]
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │  2. Glow Detection       │    │  3. FFA-Net Inference    │
        │     HSV V-channel        │    │     512×512, 3 groups    │
        │     → threshold → dilate │    │     × 19 attention blocks│
        │     → connected comps    │    │                          │
        └────────────┬─────────────┘    └────────────┬─────────────┘
                     │ glow mask +                   │ deep dehazed
                     │ per-region atm light          │ estimate
                     └───────────────┬───────────────┘
                                     ▼
                    ┌──────────────────────────────────────┐
                    │  4. Atmospheric Light + Recovery     │
                    │     dark channel (glow excluded)     │
                    │     → A_global + per-pixel A map     │
                    │     → invert scattering model        │
                    │     → glow-aware blend               │
                    └──────────────────┬───────────────────┘
                                       │  uint8 BGR
                    ┌──────────────────▼───────────────────┐
                    │  5. Post-processing                  │
                    │     CLAHE → denoise → unsharp        │
                    │     → saturation boost               │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │  6. Quality Assessment               │
                    │     PSNR, SSIM, NIQE, BRISQUE,       │
                    │     visibility, colorfulness         │
                    └──────────────────┬───────────────────┘
                                       ▼
                              DehazeResponse (JSON)
                       4 × base64 PNG + metrics + timings
```

### Colour / dtype contract

> **All pipeline-internal images are `float32` BGR in `[0,1]`. `uint8` only at
> I/O boundaries.**

Decoding an upload and encoding the response are the only places `uint8`
appears. Every service documents this contract in its module docstring. Stages
4→5 cross back to `uint8` because the post-processing operators (CLAHE,
`fastNlMeansDenoisingColored`) are defined on 8-bit images.

---

## 2. Stage descriptions

### Stage 1 — Preprocessing (`services/preprocessor.py`)

1. **Validate** — rejects non-arrays, wrong dimensionality, and images whose
   shorter side is below `min_image_dimension` (64px).
2. **Ensure BGR** — grayscale is promoted to 3-channel; BGRA has alpha dropped.
3. **Resize** — images longer than `max_image_dimension` (2048px) are downscaled
   with aspect preserved. This is a *processing cap*, not a rejection: a
   4032×3024 phone photo is accepted and downscaled.
4. **Adaptive denoising** — the Laplacian variance is used as a noise/blur proxy.
   Low variance implies a noisy or soft frame, so filtering is applied
   proportionally: strong bilateral (`d=9`) below 50, mild (`d=5`) below 200,
   none above.
5. **Selective gamma lift** — pixels with luminance below 30/255 are brightened
   with `γ = 0.7` and blended back, recovering shadow detail without blowing out
   the already-bright light sources.
6. **Normalise** to `float32 [0,1]`.

### Stage 2 — Glow detection (`services/glow_detector.py`)

Classical computer vision — no learning. Artificial lights are found by
thresholding the blurred HSV *value* channel at 85% of its peak, then dilating
with a 41×41 elliptical kernel to capture the halo. Connected components with
area ≥ 50px become `GlowRegion`s. For each region, the **local atmospheric
light** is the mean of the brightest 0.1% of pixels in a 2× expanded ROI.

A pitch-dark frame short-circuits to an empty region list and a zero mask.

### Stage 3 — FFA-Net inference (`services/ffa_net_service.py`)

The image is resized to 512×512, converted BGR→RGB, arranged as `1×3×H×W`, and
run through FFA-Net under `torch.no_grad()`. The output is converted back to BGR
and resized to the original resolution. See [model_info.md](model_info.md).

### Stage 4 — Atmospheric light + radiance recovery

See §3 for the formulas.

### Stage 5 — Post-processing (`services/postprocessor.py`)

CLAHE on the LAB *L* channel (local contrast without global brightening) →
`fastNlMeansDenoisingColored` (h=3) → unsharp mask (α=1.5, β=−0.5) → HSV
saturation boost (×1.2), which counteracts the desaturation typical of
nighttime dehazing.

### Stage 6 — Quality assessment (`services/quality_assessor.py`)

Full-reference (PSNR, SSIM) and no-reference (BRISQUE, simplified NIQE) metrics,
plus visibility and colorfulness. The *original* used as the reference is the
image **after** the Stage-1 resize, so the two frames always share identical
dimensions — a prerequisite for PSNR/SSIM.

---

## 3. The physics

### Atmospheric scattering model

The standard hazy-image formation model:

```
I(x) = J(x)·t(x) + A·(1 − t(x))
```

| Symbol | Meaning |
| ------ | ------- |
| `I(x)` | observed (hazy) intensity at pixel `x` |
| `J(x)` | true scene radiance — what we want to recover |
| `t(x)` | transmission: fraction of light reaching the camera unscattered |
| `A`    | atmospheric light (the "airlight" colour) |

Inverting for `J`:

```
J(x) = (I(x) − A) / max(t(x), t₀) + A
```

`t₀` (= `transmission_min_clip`, 0.1) floors the denominator: as `t → 0` the
division explodes and amplifies noise without it.

### Dark channel and transmission

The dark channel of a patch Ω(x):

```
J_dark(x) = min      ( min      J_c(y) )
            y∈Ω(x)     c∈{R,G,B}
```

Transmission is estimated from it, attenuated by `ω` (= `omega`, 0.95, which
retains a little haze so the result looks natural rather than synthetic):

```
t(x) = 1 − ω · I_dark(x)
```

### Per-region atmospheric light

Instead of a single global `A`, NightHaze builds a **per-pixel map**
`A(x) ∈ ℝ^{H×W×3}`:

- `A_global` — from the brightest pixels of the dark channel, computed with the
  glow regions **masked out**.
- Inside each glow region, `A` is blended from that region's `local_atm_light`
  toward `A_global` with a radial weight:

```
w(x) = clip(1 − ‖x − centre‖ / radius, 0, 1)
A(x) = w(x)·A_local + (1 − w(x))·A_global
```

### Glow-aware blending

Physics recovery over-brightens near light sources, where the model's
assumptions break down. The final image trusts the network more inside glow:

```
J_final = J_physics·(1 − 0.7·mask) + J_FFA·(0.7·mask)
```

---

## 4. Design decisions

### Why per-region atmospheric light?

The single most important departure from daytime dehazing. Daytime methods
assume one illuminant (the sun) and therefore a globally constant `A`. At night
the scene is lit by many artificial sources — sodium lamps, LEDs, neon,
headlights — each with a different colour temperature and intensity. A single
global `A` averages them into a colour that is wrong everywhere. Estimating `A`
locally around each detected light, and blending radially back to a global
estimate, keeps colours plausible across the whole frame.

### Why exclude glow regions from the dark channel?

The Dark Channel Prior assumes most local patches contain at least one very dark
colour channel. That is true for daytime outdoor scenes and **false** around
bright artificial lights, where every channel saturates. Feeding those patches
into the DCP corrupts the estimate of both `A` and `t`. Masking them out is what
keeps the prior usable at night at all.

### Why blend back toward FFA-Net inside glow?

Even with a local `A`, the inversion `(I − A)/t` is numerically hostile near
light sources: `I ≈ A` there, so the numerator is near zero while `t` is small.
The learned model has no such singularity, so it is the more trustworthy
estimate exactly where the physics is weakest. 70% is an empirical weight.

### Why pretrained weights instead of training from scratch?

1. **No paired nighttime haze dataset exists at useful scale.** Supervised
   dehazing needs (hazy, clean) pairs of the *same* scene; those are effectively
   impossible to capture at night and synthetic pairs do not transfer well.
2. **FFA-Net's learned priors are illumination-agnostic.** Channel and pixel
   attention learn *where and in which channels haze concentrates* — structure
   that transfers from the RESIDE indoor set to night scenes.
3. **The nighttime-specific problems are physical, not perceptual**, and are
   better handled by the explicit model in Stage 4 than by asking a network to
   learn them from data we do not have.

This hybrid split — learned appearance prior + explicit physics for illumination
— is the core thesis of the project.

### Why is FFA-Net run at a fixed 512×512?

FFA-Net is fully convolutional and would accept any size divisible by 16, but a
fixed input bounds inference cost (the dominant term, ~10s on CPU) independently
of upload resolution, and matches the scale the weights were trained at. The
output is resized back, so the response always matches the input dimensions.
