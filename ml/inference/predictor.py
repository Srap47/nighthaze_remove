"""
Standalone FFA-Net inference CLI.

Runs the pretrained dehazing model on a single image with no FastAPI/web
dependencies — useful for batch processing, debugging, and demonstrating that
the model works independently of the application.

Usage:
    python ml/inference/predictor.py --input path/to/hazy.jpg
    python ml/inference/predictor.py --input hazy.jpg --output clean.jpg --device cpu

Arguments:
    --input    Path to the hazy nighttime image (required).
    --output   Where to save the result (default: dehazed_<input_name>.jpg).
    --weights  Path to the .pkl checkpoint (default: ml/weights/its_train_ffa_3_19.pkl).
    --device   'cpu' or 'cuda' (default: auto-detect).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Make the `ml` package importable when this file is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model.ffa_net import FFA  # noqa: E402

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "its_train_ffa_3_19.pkl"
INPUT_SIZE = 512
GPS = 3
BLOCKS = 19


def load_model(weights_path: Path, device: str) -> torch.nn.Module:
    """Build FFA-Net and load the pretrained checkpoint."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Download "
            "'its_train_ffa_3_19.pkl' from https://github.com/zhilin007/FFA-Net"
        )

    model = FFA(gps=GPS, blocks=BLOCKS)
    checkpoint = torch.load(str(weights_path), map_location=device)

    # Training snapshot: weights live under 'model' and carry a 'module.'
    # prefix from nn.DataParallel.
    raw_state = checkpoint["model"]
    state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def dehaze(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    """Run FFA-Net on a float32 [0,1] BGR image; return float32 [0,1] BGR."""
    height, width = image.shape[:2]

    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(tensor)

    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_bgr = np.clip(output_np[:, :, ::-1].copy(), 0.0, 1.0)

    restored = cv2.resize(output_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return restored.astype(np.float32)


def postprocess(image: np.ndarray) -> np.ndarray:
    """Basic polish: CLAHE on the L channel + unsharp mask. uint8 BGR in/out."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    image = cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dehaze a nighttime image with pretrained FFA-Net."
    )
    parser.add_argument("--input", required=True, type=Path, help="Hazy input image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: dehazed_<input_name>.jpg).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"FFA-Net checkpoint (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Compute device (default: cuda if available, else cpu).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output or args.input.with_name(f"dehazed_{args.input.name}")

    if not args.input.exists():
        print(f"error: input image not found: {args.input}", file=sys.stderr)
        return 1

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not decode image: {args.input}", file=sys.stderr)
        return 1

    print(f"Device : {device}")
    print(f"Weights: {args.weights}")
    print(f"Input  : {args.input}  ({image.shape[1]}x{image.shape[0]})")

    started = time.perf_counter()

    model = load_model(args.weights, device)
    normalized = image.astype(np.float32) / 255.0
    dehazed = dehaze(model, normalized, device)
    result = postprocess(np.clip(dehazed * 255.0, 0, 255).astype(np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        print(f"error: failed to write output: {output_path}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    print(f"Output : {output_path}")
    print(f"Time   : {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
