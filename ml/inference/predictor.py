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

# Default path to pretrained weights (relative to this file's location)
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "its_train_ffa_3_19.pkl"
# TWEAK NOTE: INPUT_SIZE controls the square resolution input to FFA-Net
# Must match training resolution (512 common; 256/1024 also possible).
# Larger = more detail but slower; smaller = faster but less fine detail.
INPUT_SIZE = 512
# TWEAK NOTE: GPS (gate/group branches) and BLOCKS (units per group) must match weights
# These are hardcoded for compatibility with pretrained checkpoint (3, 19).
GPS = 3
BLOCKS = 19


def load_model(weights_path: Path, device: str) -> torch.nn.Module:
    """Build FFA-Net architecture and load pretrained weights.

    Constructs the model, loads the checkpoint, handles DataParallel naming quirks,
    and prepares for inference (eval mode, correct device).

    Args:
        weights_path: Path to the pretrained .pkl checkpoint
        device: 'cpu' or 'cuda' for inference

    Returns:
        Initialized FFA-Net model in eval mode on the specified device

    Raises:
        FileNotFoundError: If weights file doesn't exist
        RuntimeError: If checkpoint loading fails (architecture mismatch)
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Download "
            "'its_train_ffa_3_19.pkl' from https://github.com/zhilin007/FFA-Net"
        )

    # Build model with same architecture as weights were trained with
    model = FFA(gps=GPS, blocks=BLOCKS)
    # Load checkpoint (maps to specified device, avoids GPU memory issues if loading on CPU)
    checkpoint = torch.load(str(weights_path), map_location=device)

    # Training snapshot: weights live under 'model' key and carry a 'module.'
    # prefix from nn.DataParallel training (strips the prefix for single-GPU inference).
    raw_state = checkpoint["model"]
    state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}

    # Load weights into model; strict=True ensures exact architecture match
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
    model.to(device)  # Move model to compute device
    return model


def dehaze(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    """Run FFA-Net inference on a float32 [0,1] BGR image.

    Resizes input to model's expected square dimensions, runs inference,
    and resizes output back to original dimensions (maintains aspect ratio).

    Args:
        model: Initialized FFA-Net model in eval mode
        image: Float32 BGR image normalized to [0,1] range
        device: 'cpu' or 'cuda' (must match model device)

    Returns:
        Dehazed float32 BGR image [0,1] range, same dimensions as input
    """
    height, width = image.shape[:2]

    # Resize to model's input size (model expects square INPUT_SIZE x INPUT_SIZE)
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    # Convert BGR → RGB (PyTorch models typically expect RGB)
    rgb = resized[:, :, ::-1].copy()
    # Convert to tensor: HWC → CHW (channel dimension first), add batch dimension, float32
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # Inference: disable gradient computation for efficiency (not training)
    with torch.no_grad():
        output = model(tensor)

    # Convert output tensor to numpy: squeeze batch dim, CHW → HWC, RGB → BGR
    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Clip to [0,1] range (model may slightly exceed due to numerical precision)
    output_bgr = np.clip(output_np[:, :, ::-1].copy(), 0.0, 1.0)

    # Resize back to original image dimensions (linear interpolation for smooth downsampling)
    restored = cv2.resize(output_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return restored.astype(np.float32)


def postprocess(image: np.ndarray) -> np.ndarray:
    """Apply final enhancement: local contrast + sharpening.

    Polishes the dehazed image with CLAHE (brightens dark regions while
    avoiding halos) and unsharp masking (enhances edges and details).

    Args:
        image: uint8 BGR image (output of dehaze, denormalized to [0,255])

    Returns:
        Enhanced uint8 BGR image [0,255] range, same dimensions as input
    """
    # Convert to LAB color space (work in brightness/luminance domain)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    # CLAHE: local contrast enhancement on brightness channel only
    # Prevents desaturation and halo artifacts that global histogram equalization causes
    # clipLimit (2.0) controls contrast amplification; tileGridSize (8,8) defines local window
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    # Merge enhanced brightness with original color channels and convert back to BGR
    image = cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    # Unsharp masking: high-pass sharpening via blurred subtraction
    # Technique: output = original * 1.5 - blurred * 0.5 = original + 0.5 * (original - blurred)
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
    """Main entry point: dehaze an image with FFA-Net CLI.

    Orchestrates the full pipeline:
    1. Parse arguments and validate inputs
    2. Auto-detect device (GPU if available, else CPU)
    3. Load model and image
    4. Run dehaze inference
    5. Apply postprocessing
    6. Write output and report timing

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None)

    Returns:
        Exit code: 0 on success, 1 on error
    """
    args = parse_args(argv)

    # Auto-detect compute device (CUDA GPU preferred, fallback to CPU)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Set output path: use specified path or auto-generate from input name
    output_path = args.output or args.input.with_name(f"dehazed_{args.input.name}")

    # Validate inputs exist and are readable
    if not args.input.exists():
        print(f"error: input image not found: {args.input}", file=sys.stderr)
        return 1

    # Load image from disk
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not decode image: {args.input}", file=sys.stderr)
        return 1

    # Print configuration summary
    print(f"Device : {device}")
    print(f"Weights: {args.weights}")
    print(f"Input  : {args.input}  ({image.shape[1]}x{image.shape[0]})")

    # Time the entire pipeline (model load + inference + postprocess)
    started = time.perf_counter()

    # Pipeline: load → normalize → dehaze → postprocess
    model = load_model(args.weights, device)
    normalized = image.astype(np.float32) / 255.0  # Convert uint8 [0,255] to float [0,1]
    dehazed = dehaze(model, normalized, device)     # FFA-Net inference
    result = postprocess(np.clip(dehazed * 255.0, 0, 255).astype(np.uint8))  # Polish output

    # Write result to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        print(f"error: failed to write output: {output_path}", file=sys.stderr)
        return 1

    # Report success and timing
    elapsed = time.perf_counter() - started
    print(f"Output : {output_path}")
    print(f"Time   : {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    # Run as standalone script: exit with status code from main()
    raise SystemExit(main())
