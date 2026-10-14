#!/usr/bin/env python3
"""Satellite tile enhancement pipeline.

Implements classical spatial enhancement (NL-means denoising, CIELAB stretch,
CLAHE, and unsharp masking), bicubic 2x upscaling, and pretrained learned
super-resolution (FSRCNN / EDSR).
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

DENOISE_H = 3
DENOISE_H_COLOR = 3
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

CONTRAST_LOW_PERCENTILE = 1.0
CONTRAST_HIGH_PERCENTILE = 99.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

UNSHARP_SIGMA = 1.0
UNSHARP_AMOUNT = 1.0
UNSHARP_THRESHOLD = 3

DEFAULT_SR_MODEL = "fsrcnn"
DEFAULT_SR_SCALE = 2
MIN_MODEL_SIZE_BYTES = 10240

SR_MODEL_URLS = {
    "fsrcnn": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    "edsr": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
}

SR_MODEL_FILES = {
    "fsrcnn": Path("models/FSRCNN_x2.pb"),
    "edsr": Path("models/EDSR_x2.pb"),
}

_SR_CACHE: dict[str, cv2.dnn_superres.DnnSuperResImpl] = {}


def check_dnn_superres() -> None:
    if not hasattr(cv2, "dnn_superres"):
        sys.exit(
            "Error: cv2.dnn_superres is not available. Please uninstall opencv-python and install opencv-contrib-python:\n"
            "    pip uninstall -y opencv-python\n"
            "    pip install opencv-contrib-python"
        )


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def collect_images(dir_path: str | Path) -> list[Path]:
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {dir_path}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in dir_p.iterdir() if p.suffix.lower() in exts)


def denoise(bgr: np.ndarray) -> np.ndarray:
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
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(l_channel)


def unsharp(
    bgr: np.ndarray,
    sigma: float = UNSHARP_SIGMA,
    amount: float = UNSHARP_AMOUNT,
    threshold: int = UNSHARP_THRESHOLD,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigma)
    diff = bgr.astype(np.float32) - blurred.astype(np.float32)
    diff[np.abs(diff) < threshold] = 0.0
    sharpened = bgr.astype(np.float32) + amount * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_classical(bgr: np.ndarray) -> np.ndarray:
    denoised = denoise(bgr)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_enhanced = apply_clahe(stretch_levels(l_ch))
    bgr_contrast = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    return unsharp(bgr_contrast)


def upscale_bicubic(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def ensure_model_file(model_name: str) -> Path:
    key = model_name.lower()
    if key not in SR_MODEL_FILES:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(SR_MODEL_FILES.keys())}")
    target_path = SR_MODEL_FILES[key]
    if target_path.is_file() and target_path.stat().st_size >= MIN_MODEL_SIZE_BYTES:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = SR_MODEL_URLS[key]
    print(f"Downloading model '{key}' from {url}...")
    try:
        urllib.request.urlretrieve(url, str(target_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc

    if not target_path.is_file() or target_path.stat().st_size < MIN_MODEL_SIZE_BYTES:
        actual_size = target_path.stat().st_size if target_path.is_file() else 0
        raise RuntimeError(f"Downloaded model '{target_path}' is corrupted ({actual_size} bytes).")
    return target_path


def get_sr_model(model_name: str = DEFAULT_SR_MODEL) -> cv2.dnn_superres.DnnSuperResImpl:
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
    return get_sr_model(model).upsample(bgr)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "classical": enhance_classical,
    "bicubic": upscale_bicubic,
    "sr": upscale_sr,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Satellite Tile Enhancement CLI")
    parser.add_argument("--input", type=Path, required=True, help="Input tile directory or image path.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save enhanced images.")
    parser.add_argument(
        "--method",
        choices=["classical", "bicubic", "sr"],
        default="classical",
        help="Enhancement method (default: classical).",
    )
    parser.add_argument(
        "--sr-model",
        choices=["fsrcnn", "edsr"],
        default="fsrcnn",
        help="Model architecture for super-resolution (default: fsrcnn).",
    )
    return parser.parse_args()


def main() -> None:
    check_dnn_superres()
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.is_dir():
        image_paths = collect_images(args.input)
    elif args.input.is_file():
        image_paths = [args.input]
    else:
        sys.exit(f"Error: input path '{args.input}' does not exist.")

    if not image_paths:
        sys.exit(f"Error: no images found at '{args.input}'.")

    if args.method == "sr":
        fn = functools.partial(upscale_sr, model=args.sr_model)
        get_sr_model(args.sr_model)
    else:
        fn = METHODS[args.method]

    print(f"Processing {len(image_paths)} images using method: '{args.method}'...")
    total_time = 0.0

    for path in tqdm(image_paths, desc=f"Enhancing ({args.method})"):
        img = load_image(path)
        t0 = time.perf_counter()
        result = fn(img)
        total_time += time.perf_counter() - t0

        cv2.imwrite(str(args.output / f"{path.stem}.png"), result)

    avg_time = total_time / len(image_paths) if image_paths else 0.0
    print(f"Processed {len(image_paths)} images in {total_time:.3f} s ({avg_time:.4f} s/tile).")


if __name__ == "__main__":
    main()
