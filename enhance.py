#!/usr/bin/env python3
"""Satellite Tile Enhancement pipeline.

Supports classical enhancement (denoise, tone stretch, CLAHE, unsharp masking),
bicubic 2x upscaling, and pretrained learned super-resolution (FSRCNN / EDSR 2x).
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys
import time
import urllib.request
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

# Denoising constants
DENOISE_H: int = 3
DENOISE_H_COLOR: int = 3
DENOISE_TEMPLATE_WINDOW: int = 7
DENOISE_SEARCH_WINDOW: int = 21

# Contrast & Tone constants
CONTRAST_LOW_PERCENTILE: float = 1.0
CONTRAST_HIGH_PERCENTILE: float = 99.0
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

# Sharpening constants
UNSHARP_SIGMA: float = 1.0
UNSHARP_AMOUNT: float = 1.0
UNSHARP_THRESHOLD: int = 3

# Super-Resolution constants
DEFAULT_SR_MODEL: str = "fsrcnn"
DEFAULT_SR_SCALE: int = 2
MIN_MODEL_SIZE_BYTES: int = 10240

SR_MODEL_URLS: dict[str, str] = {
    "fsrcnn": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    "edsr": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
}

SR_MODEL_FILES: dict[str, Path] = {
    "fsrcnn": Path("models/FSRCNN_x2.pb"),
    "edsr": Path("models/EDSR_x2.pb"),
}

_SR_CACHE: dict[str, cv2.dnn_superres.DnnSuperResImpl] = {}


def check_dnn_superres() -> None:
    """Verify that cv2.dnn_superres is available in current OpenCV installation."""
    if not hasattr(cv2, "dnn_superres"):
        sys.exit(
            "Error: cv2.dnn_superres is not available. Please uninstall opencv-python and install opencv-contrib-python:\n"
            "    pip uninstall -y opencv-python\n"
            "    pip install opencv-contrib-python"
        )


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a 3-channel BGR uint8 numpy array."""
    path_str = str(path)
    img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {path_str}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def collect_images(dir_path: str | Path) -> list[Path]:
    """Collect image paths (PNG, JPG, JPEG) from a directory, sorted by filename."""
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {dir_path}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in dir_p.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def denoise(bgr: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means color denoising to reduce sensor noise and compression artifacts."""
    return cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=DENOISE_H,
        hColor=DENOISE_H_COLOR,
        templateWindowSize=DENOISE_TEMPLATE_WINDOW,
        searchWindowSize=DENOISE_SEARCH_WINDOW,
    )


def stretch_levels(
    l_channel: np.ndarray,
    low_pct: float = CONTRAST_LOW_PERCENTILE,
    high_pct: float = CONTRAST_HIGH_PERCENTILE,
) -> np.ndarray:
    """Stretch luminance levels between given percentiles to expand dynamic range and cut haze."""
    low_val = float(np.percentile(l_channel, low_pct))
    high_val = float(np.percentile(l_channel, high_pct))
    if high_val <= low_val:
        return l_channel.copy()
    stretched = (l_channel.astype(np.float32) - low_val) / (high_val - low_val) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_clahe(
    l_channel: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on luminance."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(l_channel)


def unsharp(
    bgr: np.ndarray,
    sigma: float = UNSHARP_SIGMA,
    amount: float = UNSHARP_AMOUNT,
    threshold: int = UNSHARP_THRESHOLD,
) -> np.ndarray:
    """Apply unsharp masking with Gaussian blur and a coring threshold to avoid noise amplification."""
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigma)
    diff = bgr.astype(np.float32) - blurred.astype(np.float32)
    mask = np.abs(diff) < threshold
    diff[mask] = 0.0
    sharpened = bgr.astype(np.float32) + amount * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_classical(bgr: np.ndarray) -> np.ndarray:
    """Run classical enhancement pipeline: denoise -> contrast stretch & CLAHE on L -> unsharp."""
    denoised = denoise(bgr)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_stretched = stretch_levels(l_ch)
    l_clahe = apply_clahe(l_stretched)
    lab_enhanced = cv2.merge([l_clahe, a_ch, b_ch])
    bgr_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return unsharp(bgr_contrast)


def upscale_bicubic(bgr: np.ndarray) -> np.ndarray:
    """Upscale image 2x using bicubic interpolation."""
    return cv2.resize(bgr, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def ensure_model_file(model_name: str) -> Path:
    """Ensure super-resolution model file exists, auto-downloading with urllib if missing."""
    key = model_name.lower()
    if key not in SR_MODEL_FILES:
        raise ValueError(f"Unsupported SR model '{model_name}'. Allowed: {list(SR_MODEL_FILES.keys())}")
    target_path = SR_MODEL_FILES[key]
    if target_path.is_file() and target_path.stat().st_size >= MIN_MODEL_SIZE_BYTES:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = SR_MODEL_URLS[key]
    print(f"Downloading super-resolution model '{key}' from {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, str(target_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc

    if not target_path.is_file() or target_path.stat().st_size < MIN_MODEL_SIZE_BYTES:
        actual_size = target_path.stat().st_size if target_path.is_file() else 0
        raise RuntimeError(
            f"Downloaded model file '{target_path}' is invalid or under 10 KB ({actual_size} bytes)."
        )
    return target_path


def get_sr_model(model_name: str = DEFAULT_SR_MODEL) -> cv2.dnn_superres.DnnSuperResImpl:
    """Retrieve or initialize cached DnnSuperResImpl instance for given model."""
    check_dnn_superres()
    key = model_name.lower()
    if key not in _SR_CACHE:
        model_path = ensure_model_file(key)
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel(key, DEFAULT_SR_SCALE)
        _SR_CACHE[key] = sr
    return _SR_CACHE[key]


def upscale_sr(bgr: np.ndarray, model: str = DEFAULT_SR_MODEL) -> np.ndarray:
    """Upscale image 2x using pretrained learned super-resolution model (cached)."""
    sr = get_sr_model(model)
    return sr.upsample(bgr)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "classical": enhance_classical,
    "bicubic": upscale_bicubic,
    "sr": upscale_sr,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Satellite Tile Enhancement CLI")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing satellite tiles or single image path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory to save enhanced PNG tiles.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["classical", "bicubic", "sr"],
        default="classical",
        help="Enhancement method: classical, bicubic, or sr.",
    )
    parser.add_argument(
        "--sr-model",
        type=str,
        choices=["fsrcnn", "edsr"],
        default="fsrcnn",
        help="Learned super-resolution model (default: fsrcnn).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    check_dnn_superres()
    args = parse_args()

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        image_paths = collect_images(input_path)
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        sys.exit(f"Error: input path '{input_path}' does not exist.")

    if not image_paths:
        sys.exit(f"Error: no images found at '{input_path}'.")

    if args.method == "sr":
        fn = functools.partial(upscale_sr, model=args.sr_model)
        # Pre-load/cache the model once before processing loop
        get_sr_model(args.sr_model)
    else:
        fn = METHODS[args.method]

    print(f"Processing {len(image_paths)} images using method: '{args.method}'...")
    total_time = 0.0

    for path in tqdm(image_paths, desc=f"Enhancing ({args.method})"):
        img = load_image(path)
        t0 = time.perf_counter()
        result = fn(img)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        out_path = output_dir / f"{path.stem}.png"
        cv2.imwrite(str(out_path), result)

    avg_time = total_time / len(image_paths) if image_paths else 0.0
    print(f"Processed {len(image_paths)} images in {total_time:.3f} s.")
    print(f"Average seconds per image: {avg_time:.4f} s.")


if __name__ == "__main__":
    main()
