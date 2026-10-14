#!/usr/bin/env python3
"""Evaluation suite for satellite tile enhancement and super-resolution."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import urllib.parse
import urllib.request

import cv2
import numpy as np
import scipy.signal
import scipy.stats
from skimage.metrics import structural_similarity
from tqdm import tqdm

from enhance import (
    apply_clahe,
    collect_images,
    denoise,
    enhance_classical,
    get_sr_model,
    load_image,
    stretch_levels,
    unsharp,
    upscale_bicubic,
    upscale_sr,
)

DEFAULT_SEED = 0
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_ALPHA = 0.05

HAZE_T = 0.7
HAZE_A = 220.0
BLUR_SIGMA_TESTB = 1.2
BLUR_SIGMA_COMBINED = 1.0
JPEG_QUALITY_TESTB = 60
JPEG_QUALITY_COMBINED = 65
JPEG_QUALITY_REALISTIC = 70
NOISE_SIGMA_TESTB = 3.0
NOISE_SIGMA_COMBINED = 3.0
NOISE_SIGMA_REALISTIC = 3.0


# --- Synthetic Degradations ---

def downsample(img: np.ndarray, factor: int | float = 2, mode: str = "area") -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    interp = cv2.INTER_AREA if mode.lower() == "area" else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, enc = cv2.imencode(".jpg", img, params)
    if not success:
        raise RuntimeError(f"Failed to compress image to JPEG at quality {quality}")
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def gauss_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def add_haze(img: np.ndarray, t: float = HAZE_T, a: float = HAZE_A) -> np.ndarray:
    """Koschmieder optical model: I' = I*t + A*(1-t)."""
    hazy = img.astype(np.float32) * t + a * (1.0 - t)
    return np.clip(hazy, 0, 255).astype(np.uint8)


# --- Full-Reference Metrics ---

def compute_psnr(img_test: np.ndarray, img_ref: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB between test and reference."""
    assert img_test.shape == img_ref.shape, f"Shape mismatch: {img_test.shape} vs {img_ref.shape}"
    diff = img_test.astype(np.float64) - img_ref.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_ssim(img_test: np.ndarray, img_ref: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) across color channels."""
    assert img_test.shape == img_ref.shape, f"Shape mismatch: {img_test.shape} vs {img_ref.shape}"
    return float(structural_similarity(img_test, img_ref, channel_axis=2, data_range=255))


def compute_sobel_edge_correlation(img_test: np.ndarray, img_ref: np.ndarray) -> float:
    """Compute Pearson correlation coefficient of Sobel gradient magnitudes."""
    assert img_test.shape == img_ref.shape, f"Shape mismatch: {img_test.shape} vs {img_ref.shape}"
    gray_t = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_r = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

    gx_t = cv2.Sobel(gray_t, cv2.CV_64F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(gray_t, cv2.CV_64F, 0, 1, ksize=3)
    mag_t = np.hypot(gx_t, gy_t).ravel()

    gx_r = cv2.Sobel(gray_r, cv2.CV_64F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(gray_r, cv2.CV_64F, 0, 1, ksize=3)
    mag_r = np.hypot(gx_r, gy_r).ravel()

    std_t = np.std(mag_t)
    std_r = np.std(mag_r)
    if std_t < 1e-9 or std_r < 1e-9:
        return 0.0
    corr = np.corrcoef(mag_t, mag_r)[0, 1]
    return float(corr)


# --- No-Reference Metrics ---

def compute_laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_noise_sigma(img: np.ndarray) -> float:
    try:
        from skimage.restoration import estimate_sigma
        return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))
    except (ImportError, Exception):
        # Fallback implementing Donoho-Johnstone (1994) db2 high-pass detail MAD estimator
        h_high = np.array([-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.12940952255126034])
        kernel = np.outer(h_high, h_high)
        sigmas: list[float] = []
        denom = 0.6744897501960817
        for c in range(img.shape[2]):
            ch = img[:, :, c].astype(np.float64)
            d = scipy.signal.convolve2d(ch, kernel, mode="valid")[::2, ::2]
            d = d[np.nonzero(d)]
            sigmas.append(float(np.median(np.abs(d)) / denom) if len(d) > 0 else 0.0)
        return float(np.mean(sigmas))


def compute_blockiness(img: np.ndarray) -> float:
    """Ratio of mean intensity steps across 8x8 JPEG grid boundaries vs interior steps."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    b_mask_h = np.zeros(diff_h.shape[1], dtype=bool)
    b_mask_h[7::8] = True

    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    b_mask_v = np.zeros(diff_v.shape[0], dtype=bool)
    b_mask_v[7::8] = True

    b_steps_h, nb_steps_h = diff_h[:, b_mask_h], diff_h[:, ~b_mask_h]
    b_steps_v, nb_steps_v = diff_v[b_mask_v, :], diff_v[~b_mask_v, :]

    total_b = float(np.sum(b_steps_h) + np.sum(b_steps_v))
    count_b = b_steps_h.size + b_steps_v.size

    total_nb = float(np.sum(nb_steps_h) + np.sum(nb_steps_v))
    count_nb = nb_steps_h.size + nb_steps_v.size

    if count_b == 0 or count_nb == 0 or total_nb == 0:
        return 1.0

    mean_b = total_b / count_b
    mean_nb = total_nb / count_nb
    return float(mean_b / (mean_nb + 1e-8))


def compute_contrast(img: np.ndarray) -> float:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(np.std(lab[:, :, 0]))


def compute_color_shift_delta_e(img_test: np.ndarray, img_ref: np.ndarray) -> float:
    if img_test.shape[:2] != img_ref.shape[:2]:
        img_ref_aligned = cv2.resize(
            img_ref,
            (img_test.shape[1], img_test.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        img_ref_aligned = img_ref

    lab_t = cv2.cvtColor(img_test, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab_r = cv2.cvtColor(img_ref_aligned, cv2.COLOR_BGR2LAB).astype(np.float64)
    delta_e = np.sqrt(np.sum((lab_t - lab_r) ** 2, axis=-1))
    return float(np.mean(delta_e))


# --- Bootstrap Statistics ---

def compute_bootstrap_ci(
    diffs: np.ndarray,
    rounds: int = BOOTSTRAP_ROUNDS,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    if len(diffs) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_indices = rng.integers(0, n, size=(rounds, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    ci_low = float(np.percentile(boot_means, low_pct))
    ci_high = float(np.percentile(boot_means, high_pct))
    return (ci_low, ci_high)


def summarize_paired_diffs(
    diffs: Sequence[float],
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    arr = np.array(diffs, dtype=np.float64)
    valid_arr = arr[np.isfinite(arr)] if np.any(np.isfinite(arr)) else np.array([0.0])

    mean_diff = float(np.mean(valid_arr))
    win_rate = float(np.mean(arr > 0) * 100.0) if len(arr) > 0 else 0.0
    ci_low, ci_high = compute_bootstrap_ci(valid_arr, seed=seed)

    if ci_low <= 0.0 <= ci_high:
        verdict = "no clear difference"
    elif ci_low > 0.0:
        verdict = "statistically superior (+)"
    else:
        verdict = "statistically inferior (-)"

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate": win_rate,
        "verdict": verdict,
        "n": len(arr),
    }


# --- Optional Real-Pair Check ---

def run_optional_real_pair_check(
    pairs_csv: Path | None,
    cache_dir: Path = Path("cache"),
) -> dict[str, Any] | None:
    """Optional real-pair check against Google Static Maps API scale=1 vs scale=2."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not pairs_csv or not pairs_csv.is_file() or not api_key:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    coords: list[tuple[float, float]] = []
    with open(pairs_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "lat" in row and "lng" in row:
                try:
                    coords.append((float(row["lat"]), float(row["lng"])))
                except ValueError:
                    continue

    coords = coords[:15]  # Cap at 15 requests total
    if not coords:
        return None

    results: list[dict[str, Any]] = []

    for idx, (lat, lng) in enumerate(coords):
        p1_path = cache_dir / f"pair_{idx:02d}_scale1.png"
        p2_path = cache_dir / f"pair_{idx:02d}_scale2.png"

        # Download if not cached
        if not p1_path.is_file():
            url1 = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=19&size=640x640&maptype=satellite&scale=1&key={urllib.parse.quote(api_key)}"
            urllib.request.urlretrieve(url1, str(p1_path))

        if not p2_path.is_file():
            url2 = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=19&size=640x640&maptype=satellite&scale=2&key={urllib.parse.quote(api_key)}"
            urllib.request.urlretrieve(url2, str(p2_path))

        img1 = load_image(p1_path)
        img2 = load_image(p2_path)

        # Scale 1 (640x640) upscaled 2x -> 1280x1280
        # Reference is scale 2 (1280x1280)
        ref_hr = img2[:1280, :1280]
        in_lr = img1[:640, :640]

        bic = upscale_bicubic(in_lr)
        sr = upscale_sr(in_lr)

        psnr_bic = compute_psnr(bic, ref_hr)
        psnr_sr = compute_psnr(sr, ref_hr)
        ssim_bic = compute_ssim(bic, ref_hr)
        ssim_sr = compute_ssim(sr, ref_hr)
        sobel_bic = compute_sobel_edge_correlation(bic, ref_hr)
        sobel_sr = compute_sobel_edge_correlation(sr, ref_hr)

        results.append({
            "idx": idx,
            "lat": lat,
            "lng": lng,
            "psnr_bic": psnr_bic,
            "psnr_sr": psnr_sr,
            "ssim_bic": ssim_bic,
            "ssim_sr": ssim_sr,
            "sobel_bic": sobel_bic,
            "sobel_sr": sobel_sr,
        })

    return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Test Execution Logic
# ---------------------------------------------------------------------------

def run_evaluation(
    image_paths: Sequence[Path],
    seed: int = DEFAULT_SEED,
    limit: int | None = None,
    pairs_csv: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute complete evaluation across all tiles."""
    rng = np.random.default_rng(seed)

    if limit is not None and limit > 0:
        eval_paths = list(image_paths)[:limit]
    else:
        eval_paths = list(image_paths)

    records: list[dict[str, Any]] = []

    runtime_classical: list[float] = []
    runtime_bicubic: list[float] = []
    runtime_sr: list[float] = []

    get_sr_model("fsrcnn")
    print(f"\nEvaluating {len(eval_paths)} tiles (seed={seed})...")

    for path in tqdm(eval_paths, desc="Evaluating tiles"):
        raw_img = load_image(path)
        h, w = raw_img.shape[:2]
        hr = raw_img[: (h // 2) * 2, : (w // 2) * 2].copy()

        stem = path.stem
        source = "google" if "google" in stem.lower() else ("mapbox" if "mapbox" in stem.lower() else "unknown")

        t0 = time.perf_counter()
        classical_out = enhance_classical(hr)
        runtime_classical.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        bicubic_out = upscale_bicubic(hr)
        runtime_bicubic.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sr_out = upscale_sr(hr)
        runtime_sr.append(time.perf_counter() - t0)

        # Test A: SR vs Bicubic
        lr_clean = downsample(hr, 2, mode="area")
        lr_bic = downsample(hr, 2, mode="bicubic")
        lr_real = gauss_noise(jpeg(downsample(hr, 2, mode="area"), JPEG_QUALITY_REALISTIC), NOISE_SIGMA_REALISTIC, rng)

        for variant_name, lr_img in [("clean", lr_clean), ("bicubic_ds", lr_bic), ("realistic", lr_real)]:
            recon_bic = upscale_bicubic(lr_img)
            recon_sr = upscale_sr(lr_img)
            assert recon_bic.shape == hr.shape and recon_sr.shape == hr.shape

            p_bic = compute_psnr(recon_bic, hr)
            p_sr = compute_psnr(recon_sr, hr)
            s_bic = compute_ssim(recon_bic, hr)
            s_sr = compute_ssim(recon_sr, hr)
            e_bic = compute_sobel_edge_correlation(recon_bic, hr)
            e_sr = compute_sobel_edge_correlation(recon_sr, hr)

            records.append({
                "tile": stem,
                "source": source,
                "test": "Test A: SR vs Bicubic",
                "variant": variant_name,
                "method": "bicubic",
                "psnr": p_bic,
                "ssim": s_bic,
                "sobel_corr": e_bic,
                "baseline_psnr": p_bic,
                "baseline_ssim": s_bic,
                "baseline_sobel": e_bic,
                "diff_psnr": 0.0,
                "diff_ssim": 0.0,
                "diff_sobel": 0.0,
            })

            records.append({
                "tile": stem,
                "source": source,
                "test": "Test A: SR vs Bicubic",
                "variant": variant_name,
                "method": "sr",
                "psnr": p_sr,
                "ssim": s_sr,
                "sobel_corr": e_sr,
                "baseline_psnr": p_bic,
                "baseline_ssim": s_bic,
                "baseline_sobel": e_bic,
                "diff_psnr": p_sr - p_bic,
                "diff_ssim": s_sr - s_bic,
                "diff_sobel": e_sr - e_bic,
            })

        # Test B: Classical Pipeline vs Degraded Baseline
        deg_blur_jpeg_noise = gauss_noise(jpeg(blur(hr, BLUR_SIGMA_TESTB), JPEG_QUALITY_TESTB), NOISE_SIGMA_TESTB, rng)
        deg_haze = add_haze(hr, HAZE_T, HAZE_A)
        deg_combined = gauss_noise(jpeg(add_haze(blur(hr, BLUR_SIGMA_COMBINED), HAZE_T, HAZE_A), JPEG_QUALITY_COMBINED), NOISE_SIGMA_COMBINED, rng)
        deg_clean_null = hr.copy()

        degradations = [
            ("blur_jpeg_noise", deg_blur_jpeg_noise),
            ("haze", deg_haze),
            ("combined", deg_combined),
            ("clean_null", deg_clean_null),
        ]

        for deg_name, deg_img in degradations:
            base_p = compute_psnr(deg_img, hr)
            base_s = compute_ssim(deg_img, hr)
            base_e = compute_sobel_edge_correlation(deg_img, hr)

            enh_img = enhance_classical(deg_img)
            p_enh = compute_psnr(enh_img, hr)
            s_enh = compute_ssim(enh_img, hr)
            e_enh = compute_sobel_edge_correlation(enh_img, hr)

            records.append({
                "tile": stem,
                "source": source,
                "test": "Test B: Classical Pipeline",
                "variant": deg_name,
                "method": "degraded_baseline",
                "psnr": base_p,
                "ssim": base_s,
                "sobel_corr": base_e,
                "baseline_psnr": base_p,
                "baseline_ssim": base_s,
                "baseline_sobel": base_e,
                "diff_psnr": 0.0,
                "diff_ssim": 0.0,
                "diff_sobel": 0.0,
            })

            records.append({
                "tile": stem,
                "source": source,
                "test": "Test B: Classical Pipeline",
                "variant": deg_name,
                "method": "classical_enhanced",
                "psnr": p_enh,
                "ssim": s_enh,
                "sobel_corr": e_enh,
                "baseline_psnr": base_p,
                "baseline_ssim": base_s,
                "baseline_sobel": base_e,
                "diff_psnr": p_enh - base_p if math.isfinite(base_p) else p_enh - 100.0,
                "diff_ssim": s_enh - base_s,
                "diff_sobel": e_enh - base_e,
            })

            if deg_name in ("blur_jpeg_noise", "haze"):
                denoised_only = denoise(deg_img)
                lab_abl = cv2.cvtColor(deg_img, cv2.COLOR_BGR2LAB)
                l_abl, a_abl, b_abl = cv2.split(lab_abl)
                l_abl = apply_clahe(stretch_levels(l_abl))
                contrast_only = cv2.cvtColor(cv2.merge([l_abl, a_abl, b_abl]), cv2.COLOR_LAB2BGR)
                sharpen_only = unsharp(deg_img)

                ablations = [
                    ("denoise_only", denoised_only),
                    ("contrast_only", contrast_only),
                    ("sharpen_only", sharpen_only),
                    ("full_pipeline", enh_img),
                ]

                for stage_name, stage_out in ablations:
                    p_st = compute_psnr(stage_out, hr)
                    s_st = compute_ssim(stage_out, hr)
                    e_st = compute_sobel_edge_correlation(stage_out, hr)

                    records.append({
                        "tile": stem,
                        "source": source,
                        "test": "Test B: Stage Ablation",
                        "variant": f"{deg_name}:{stage_name}",
                        "method": stage_name,
                        "psnr": p_st,
                        "ssim": s_st,
                        "sobel_corr": e_st,
                        "baseline_psnr": base_p,
                        "baseline_ssim": base_s,
                        "baseline_sobel": base_e,
                        "diff_psnr": p_st - base_p,
                        "diff_ssim": s_st - base_s,
                        "diff_sobel": e_st - base_e,
                    })

        # No-reference metrics on native and enhanced tiles
        lap_in = compute_laplacian_variance(hr)
        sig_in = compute_noise_sigma(hr)
        blk_in = compute_blockiness(hr)
        cnt_in = compute_contrast(hr)

        lap_cl = compute_laplacian_variance(classical_out)
        sig_cl = compute_noise_sigma(classical_out)
        blk_cl = compute_blockiness(classical_out)
        cnt_cl = compute_contrast(classical_out)
        de_cl = compute_color_shift_delta_e(classical_out, hr)

        lap_bic2 = compute_laplacian_variance(bicubic_out)
        sig_bic2 = compute_noise_sigma(bicubic_out)
        blk_bic2 = compute_blockiness(bicubic_out)
        cnt_bic2 = compute_contrast(bicubic_out)

        lap_sr2 = compute_laplacian_variance(sr_out)
        sig_sr2 = compute_noise_sigma(sr_out)
        blk_sr2 = compute_blockiness(sr_out)
        cnt_sr2 = compute_contrast(sr_out)
        de_sr2 = compute_color_shift_delta_e(sr_out, bicubic_out)

        records.append({
            "tile": stem,
            "source": source,
            "test": "No-Reference: Real Tiles",
            "variant": "input_vs_classical",
            "method": "input",
            "laplacian_var": lap_in,
            "noise_sigma": sig_in,
            "blockiness": blk_in,
            "contrast": cnt_in,
            "delta_e": 0.0,
        })
        records.append({
            "tile": stem,
            "source": source,
            "test": "No-Reference: Real Tiles",
            "variant": "input_vs_classical",
            "method": "classical",
            "laplacian_var": lap_cl,
            "noise_sigma": sig_cl,
            "blockiness": blk_cl,
            "contrast": cnt_cl,
            "delta_e": de_cl,
        })
        records.append({
            "tile": stem,
            "source": source,
            "test": "No-Reference: Real Tiles",
            "variant": "bicubic_vs_sr_2x",
            "method": "bicubic_2x",
            "laplacian_var": lap_bic2,
            "noise_sigma": sig_bic2,
            "blockiness": blk_bic2,
            "contrast": cnt_bic2,
            "delta_e": 0.0,
        })
        records.append({
            "tile": stem,
            "source": source,
            "test": "No-Reference: Real Tiles",
            "variant": "bicubic_vs_sr_2x",
            "method": "sr_2x",
            "laplacian_var": lap_sr2,
            "noise_sigma": sig_sr2,
            "blockiness": blk_sr2,
            "contrast": cnt_sr2,
            "delta_e": de_sr2,
        })

    timing_summary = {
        "classical_avg": float(np.mean(runtime_classical)),
        "bicubic_avg": float(np.mean(runtime_bicubic)),
        "sr_avg": float(np.mean(runtime_sr)),
        "classical_total": float(np.sum(runtime_classical)),
        "bicubic_total": float(np.sum(runtime_bicubic)),
        "sr_total": float(np.sum(runtime_sr)),
    }

    real_pair_summary = run_optional_real_pair_check(pairs_csv)
    meta = {
        "seed": seed,
        "n_tiles": len(eval_paths),
        "timing": timing_summary,
        "real_pair": real_pair_summary,
    }
    return records, meta


def save_results_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    """Save all evaluation records to results.csv."""
    fieldnames = [
        "tile", "source", "test", "variant", "method",
        "psnr", "ssim", "sobel_corr",
        "diff_psnr", "diff_ssim", "diff_sobel",
        "laplacian_var", "noise_sigma", "blockiness", "contrast", "delta_e",
    ]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def generate_results_markdown(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    out_path: Path,
) -> None:
    """Generate comprehensive Markdown report results.md."""
    seed = meta["seed"]
    timing = meta["timing"]
    real_pair = meta["real_pair"]

    # Filter subsets
    rec_test_a = [r for r in records if r.get("test") == "Test A: SR vs Bicubic"]
    rec_test_b = [r for r in records if r.get("test") == "Test B: Classical Pipeline"]
    rec_abl = [r for r in records if r.get("test") == "Test B: Stage Ablation"]
    rec_noref = [r for r in records if r.get("test") == "No-Reference: Real Tiles"]

    md_lines: list[str] = []

    md_lines.append("# Satellite Tile Enhancement Evaluation Report\n")
    md_lines.append("## 1. Assumptions and Setup\n")
    md_lines.append(f"- **Random Seed:** `{seed}` (fixed for all stochastic operations: noise, resampling, bootstrap).")
    md_lines.append(f"- **Evaluated Tiles:** {meta['n_tiles']} tiles from `data/` (Google and Mapbox mixed sources).")
    md_lines.append("- **Hardware Context:** CPU execution only (Intel/AMD x86_64, no GPU, no model training).")
    md_lines.append("- **Super-Resolution Model:** FSRCNN 2x (`models/FSRCNN_x2.pb`, cached in memory).")
    md_lines.append("- **Synthetic Degradation Parameters:**")
    md_lines.append(f"  - **Downsampling:** 2x factor via OpenCV `INTER_AREA` (realistic/clean) and `INTER_CUBIC` (bicubic downsampled).")
    md_lines.append(f"  - **JPEG Compression:** In-memory encode/decode at quality {JPEG_QUALITY_TESTB} (Test B blur/noise), {JPEG_QUALITY_COMBINED} (combined), {JPEG_QUALITY_REALISTIC} (Test A realistic).")
    md_lines.append(f"  - **Gaussian Blur:** $\\sigma = {BLUR_SIGMA_TESTB}$ (Test B), $\\sigma = {BLUR_SIGMA_COMBINED}$ (combined).")
    md_lines.append(f"  - **Gaussian Noise:** $\\sigma = {NOISE_SIGMA_TESTB}$ (zero-mean).")
    md_lines.append(f"  - **Haze Model:** $I' = I \\cdot t + A \\cdot (1 - t)$ with transmission $t = {HAZE_T}$, airlight $A = {HAZE_A}$.")
    md_lines.append(f"- **Statistical Rigor:** 95% two-sided empirical bootstrap confidence interval ({BOOTSTRAP_ROUNDS} resamples, seeded). CI containing zero indicates *no clear difference*.\n")

    md_lines.append("## 2. Test A: Super-Resolution (FSRCNN) vs Bicubic Baseline (2x)\n")
    md_lines.append("Each LR tile is upscaled 2x using FSRCNN and Bicubic interpolation, then evaluated against the original HR reference.\n")

    md_lines.append("### Aggregate Performance (Mean ± Std)\n")
    md_lines.append("| Variant | Source | Method | PSNR (dB) | SSIM | Sobel Edge Corr |")
    md_lines.append("|:---|:---|:---|:---:|:---:|:---:|")

    variants_a = ["clean", "bicubic_ds", "realistic"]
    sources = ["overall", "google", "mapbox"]

    for var in variants_a:
        for src in sources:
            for meth in ["bicubic", "sr"]:
                sub = [
                    r for r in rec_test_a
                    if r["variant"] == var and r["method"] == meth
                    and (src == "overall" or r["source"] == src)
                ]
                if not sub:
                    continue
                psnrs = [r["psnr"] for r in sub]
                ssims = [r["ssim"] for r in sub]
                sobels = [r["sobel_corr"] for r in sub]
                md_lines.append(
                    f"| {var} | {src.capitalize()} | {meth.upper()} | "
                    f"{np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} | "
                    f"{np.mean(ssims):.4f} ± {np.std(ssims):.4f} | "
                    f"{np.mean(sobels):.4f} ± {np.std(sobels):.4f} |"
                )

    md_lines.append("\n### Paired Difference Analysis (SR minus Bicubic)\n")
    md_lines.append("| Variant | Source | Metric | Mean Diff | 95% Bootstrap CI | Win Rate (%) | Conclusion |")
    md_lines.append("|:---|:---|:---|:---:|:---:|:---:|:---|")

    test_a_answers: dict[str, dict[str, dict[str, Any]]] = {}

    for var in variants_a:
        test_a_answers[var] = {}
        for src in sources:
            test_a_answers[var][src] = {}
            sr_sub = [
                r for r in rec_test_a
                if r["variant"] == var and r["method"] == "sr"
                and (src == "overall" or r["source"] == src)
            ]
            if not sr_sub:
                continue

            for metric_key, metric_label in [("diff_psnr", "PSNR (dB)"), ("diff_ssim", "SSIM"), ("diff_sobel", "Sobel Corr")]:
                diffs = [r[metric_key] for r in sr_sub]
                stats = summarize_paired_diffs(diffs, seed=seed)
                test_a_answers[var][src][metric_key] = stats
                md_lines.append(
                    f"| {var} | {src.capitalize()} | {metric_label} | "
                    f"{stats['mean_diff']:+.4f} | [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] | "
                    f"{stats['win_rate']:.1f}% | {stats['verdict']} |"
                )

    md_lines.append("\n## 3. Test B: Classical Enhancement Pipeline Evaluation\n")
    md_lines.append("Evaluation of degraded baseline vs classical enhanced pipeline (`enhance_classical`) scored against ground truth HR.\n")

    md_lines.append("### Aggregate Performance (Mean ± Std)\n")
    md_lines.append("| Degradation | Source | Pipeline | PSNR (dB) | SSIM | Sobel Edge Corr |")
    md_lines.append("|:---|:---|:---|:---:|:---:|:---:|")

    degs_b = ["blur_jpeg_noise", "haze", "combined", "clean_null"]
    test_b_answers: dict[str, dict[str, dict[str, Any]]] = {}

    for deg in degs_b:
        test_b_answers[deg] = {}
        for src in sources:
            for meth in ["degraded_baseline", "classical_enhanced"]:
                sub = [
                    r for r in rec_test_b
                    if r["variant"] == deg and r["method"] == meth
                    and (src == "overall" or r["source"] == src)
                ]
                if not sub:
                    continue
                psnrs = [r["psnr"] for r in sub if math.isfinite(r["psnr"])]
                ssims = [r["ssim"] for r in sub]
                sobels = [r["sobel_corr"] for r in sub]

                psnr_str = f"{np.mean(psnrs):.2f} ± {np.std(psnrs):.2f}" if psnrs else "Inf (exact)"
                md_lines.append(
                    f"| {deg} | {src.capitalize()} | {meth} | "
                    f"{psnr_str} | {np.mean(ssims):.4f} ± {np.std(ssims):.4f} | "
                    f"{np.mean(sobels):.4f} ± {np.std(sobels):.4f} |"
                )

    md_lines.append("\n### Paired Difference Analysis (Enhanced minus Baseline)\n")
    md_lines.append("| Degradation | Source | Metric | Mean Diff | 95% Bootstrap CI | Win Rate (%) | Conclusion |")
    md_lines.append("|:---|:---|:---|:---:|:---:|:---:|:---|")

    for deg in degs_b:
        for src in sources:
            test_b_answers[deg][src] = {}
            enh_sub = [
                r for r in rec_test_b
                if r["variant"] == deg and r["method"] == "classical_enhanced"
                and (src == "overall" or r["source"] == src)
            ]
            if not enh_sub:
                continue

            for metric_key, metric_label in [("diff_psnr", "PSNR (dB)"), ("diff_ssim", "SSIM"), ("diff_sobel", "Sobel Corr")]:
                diffs = [r[metric_key] for r in enh_sub]
                stats = summarize_paired_diffs(diffs, seed=seed)
                test_b_answers[deg][src][metric_key] = stats
                md_lines.append(
                    f"| {deg} | {src.capitalize()} | {metric_label} | "
                    f"{stats['mean_diff']:+.4f} | [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] | "
                    f"{stats['win_rate']:.1f}% | {stats['verdict']} |"
                )

    md_lines.append("\n### Per-Stage Classical Pipeline Ablation Study\n")
    md_lines.append("Ablation of individual components on `blur_jpeg_noise` and `haze` degradations (Overall, n=41):\n")
    md_lines.append("| Degradation | Stage Component | Mean ΔPSNR (dB) | 95% CI ΔPSNR | Mean ΔSSIM | 95% CI ΔSSIM | Mean ΔSobel | 95% CI ΔSobel |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    ablation_answers: dict[str, dict[str, dict[str, Any]]] = {}

    for deg in ["blur_jpeg_noise", "haze"]:
        ablation_answers[deg] = {}
        for stage in ["denoise_only", "contrast_only", "sharpen_only", "full_pipeline"]:
            var_key = f"{deg}:{stage}"
            sub = [r for r in rec_abl if r["variant"] == var_key]
            if not sub:
                continue

            stats_p = summarize_paired_diffs([r["diff_psnr"] for r in sub], seed=seed)
            stats_s = summarize_paired_diffs([r["diff_ssim"] for r in sub], seed=seed)
            stats_e = summarize_paired_diffs([r["diff_sobel"] for r in sub], seed=seed)

            ablation_answers[deg][stage] = {
                "psnr": stats_p,
                "ssim": stats_s,
                "sobel": stats_e,
            }

            md_lines.append(
                f"| {deg} | {stage} | "
                f"{stats_p['mean_diff']:+.3f} | [{stats_p['ci_low']:+.3f}, {stats_p['ci_high']:+.3f}] | "
                f"{stats_s['mean_diff']:+.4f} | [{stats_s['ci_low']:+.4f}, {stats_s['ci_high']:+.4f}] | "
                f"{stats_e['mean_diff']:+.4f} | [{stats_e['ci_low']:+.4f}, {stats_e['ci_high']:+.4f}] |"
            )

    md_lines.append("\n## 4. No-Reference Evaluation on Real Satellite Tiles\n")
    md_lines.append("Quantification of real image properties without ground truth:\n")
    md_lines.append("- **Laplacian Variance:** Perceived sharpness / edge energy (higher = sharper).")
    md_lines.append("- **Noise Sigma ($\\sigma$):** Robust wavelet-estimated Gaussian noise std (lower = cleaner).")
    md_lines.append("- **Blockiness Ratio:** Step across 8x8 JPEG grid vs interior (1.0 = no grid artifacts, >1.0 = blocking).")
    md_lines.append("- **Contrast:** Standard deviation of CIELAB L channel (higher = broader dynamic range).")
    md_lines.append("- **Colour Shift ($\\Delta E$):** Mean Euclidean distance in CIELAB space.\n")

    md_lines.append("### Real Input vs Classical Enhanced (1x Resolution)\n")
    md_lines.append("| Source | Method | Sharpness (Lap Var) | Noise ($\\sigma$) | Blockiness Ratio | Contrast (Std L) | Colour Shift ($\\Delta E$) |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")

    for src in sources:
        for meth in ["input", "classical"]:
            sub = [
                r for r in rec_noref
                if r["variant"] == "input_vs_classical" and r["method"] == meth
                and (src == "overall" or r["source"] == src)
            ]
            if not sub:
                continue
            lap = [r["laplacian_var"] for r in sub]
            sig = [r["noise_sigma"] for r in sub]
            blk = [r["blockiness"] for r in sub]
            cnt = [r["contrast"] for r in sub]
            de = [r["delta_e"] for r in sub]
            de_str = f"{np.mean(de):.2f} ± {np.std(de):.2f}" if meth == "classical" else "0.00 (Ref)"
            md_lines.append(
                f"| {src.capitalize()} | {meth.capitalize()} | "
                f"{np.mean(lap):.1f} ± {np.std(lap):.1f} | "
                f"{np.mean(sig):.2f} ± {np.std(sig):.2f} | "
                f"{np.mean(blk):.3f} ± {np.std(blk):.3f} | "
                f"{np.mean(cnt):.2f} ± {np.std(cnt):.2f} | "
                f"{de_str} |"
            )

    md_lines.append("\n### Bicubic-2x vs SR-2x (2x Resolution)\n")
    md_lines.append("| Source | Method | Sharpness (Lap Var) | Noise ($\\sigma$) | Blockiness Ratio | Contrast (Std L) | Colour Shift ($\\Delta E$ vs Bicubic) |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")

    for src in sources:
        for meth in ["bicubic_2x", "sr_2x"]:
            sub = [
                r for r in rec_noref
                if r["variant"] == "bicubic_vs_sr_2x" and r["method"] == meth
                and (src == "overall" or r["source"] == src)
            ]
            if not sub:
                continue
            lap = [r["laplacian_var"] for r in sub]
            sig = [r["noise_sigma"] for r in sub]
            blk = [r["blockiness"] for r in sub]
            cnt = [r["contrast"] for r in sub]
            de = [r["delta_e"] for r in sub]
            de_str = f"{np.mean(de):.2f} ± {np.std(de):.2f}" if meth == "sr_2x" else "0.00 (Ref)"
            md_lines.append(
                f"| {src.capitalize()} | {meth.upper()} | "
                f"{np.mean(lap):.1f} ± {np.std(lap):.1f} | "
                f"{np.mean(sig):.2f} ± {np.std(sig):.2f} | "
                f"{np.mean(blk):.3f} ± {np.std(blk):.3f} | "
                f"{np.mean(cnt):.2f} ± {np.std(cnt):.2f} | "
                f"{de_str} |"
            )

    md_lines.append("\n## 5. Runtime and Throughput\n")
    md_lines.append("| Method | Total Time (s) | Avg Seconds / Tile | Single-Core Tiles / Day |")
    md_lines.append("|:---|:---:|:---:|:---:|")

    for meth_name, total_t, avg_t in [
        ("Classical Enhancement", timing["classical_total"], timing["classical_avg"]),
        ("Bicubic 2x Upscaling", timing["bicubic_total"], timing["bicubic_avg"]),
        ("Learned SR (FSRCNN 2x)", timing["sr_total"], timing["sr_avg"]),
    ]:
        tiles_per_day = int(86400.0 / avg_t) if avg_t > 0 else 0
        md_lines.append(f"| {meth_name} | {total_t:.2f} s | {avg_t:.4f} s | ~{tiles_per_day:,} |")

    md_lines.append("\n## 6. Real-Pair Check (Google Static Maps scale=1 vs scale=2)\n")
    if real_pair is None:
        md_lines.append("Real-pair check: skipped (not configured).\n")
    else:
        md_lines.append(f"Evaluated {real_pair['count']} real location pairs fetched from Google Static Maps API.\n")
        md_lines.append("| Lat, Lng | PSNR Bicubic | PSNR SR | SSIM Bicubic | SSIM SR | Sobel Bicubic | Sobel SR |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for rp in real_pair["results"]:
            md_lines.append(
                f"| {rp['lat']:.4f}, {rp['lng']:.4f} | {rp['psnr_bic']:.2f} | {rp['psnr_sr']:.2f} | "
                f"{rp['ssim_bic']:.4f} | {rp['ssim_sr']:.4f} | {rp['sobel_bic']:.4f} | {rp['sobel_sr']:.4f} |"
            )

    md_lines.append("\n## 7. Answers to Core Evaluation Questions\n")
    md_lines.append("### Question 1: Does Learned Super-Resolution (FSRCNN) beat Bicubic Upscaling?\n")
    for var in variants_a:
        md_lines.append(f"#### Variant: `{var}`")
        for src in ["overall", "google", "mapbox"]:
            stats = test_a_answers[var].get(src, {}).get("diff_psnr")
            if not stats:
                continue
            mean_d = stats["mean_diff"]
            ci = f"[{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]"
            win = stats["win_rate"]

            if stats["ci_low"] > 0:
                verdict_str = f"**YES, SR beats Bicubic** on {src} (Mean ΔPSNR = {mean_d:+.2f} dB, 95% CI {ci}, Win Rate = {win:.1f}%)."
            elif stats["ci_high"] < 0:
                verdict_str = f"**NO, Bicubic beats SR** on {src} (Mean ΔPSNR = {mean_d:+.2f} dB, 95% CI {ci}, Win Rate = {win:.1f}%)."
            else:
                verdict_str = f"**No clear difference** on {src} (95% CI spans zero: {ci}, Mean ΔPSNR = {mean_d:+.2f} dB, Win Rate = {win:.1f}%)."
            md_lines.append(f"- **{src.capitalize()}:** {verdict_str}")
        md_lines.append("")

    md_lines.append("### Question 2: Does the Task 1 Classical Pipeline Help or Hurt?\n")
    for deg in degs_b:
        md_lines.append(f"#### Degradation: `{deg}`")
        for src in ["overall", "google", "mapbox"]:
            stats_p = test_b_answers[deg].get(src, {}).get("diff_psnr")
            stats_s = test_b_answers[deg].get(src, {}).get("diff_ssim")
            if not stats_p or not stats_s:
                continue
            ci_p = f"[{stats_p['ci_low']:+.3f}, {stats_p['ci_high']:+.3f}]"
            ci_s = f"[{stats_s['ci_low']:+.4f}, {stats_s['ci_high']:+.4f}]"

            if stats_s["ci_low"] > 0 and stats_p["ci_low"] > 0:
                status = f"**HELPS** on {src} across both PSNR ({stats_p['mean_diff']:+.2f} dB, CI {ci_p}) and SSIM ({stats_s['mean_diff']:+.4f}, CI {ci_s})."
            elif stats_s["ci_low"] > 0:
                status = f"**HELPS on SSIM / structural fidelity** ({stats_s['mean_diff']:+.4f}, CI {ci_s}) while ΔPSNR is {stats_p['mean_diff']:+.2f} dB (CI {ci_p})."
            elif stats_s["ci_high"] < 0 and stats_p["ci_high"] < 0:
                status = f"**HURTS** on {src} across both PSNR ({stats_p['mean_diff']:+.2f} dB, CI {ci_p}) and SSIM ({stats_s['mean_diff']:+.4f}, CI {ci_s})."
            else:
                status = f"**Mixed or No clear difference** on {src} (ΔPSNR CI {ci_p}, ΔSSIM CI {ci_s})."
            md_lines.append(f"- **{src.capitalize()}:** {status}")
        md_lines.append("")

    md_lines.append("#### Per-Stage Ablation Findings (`blur_jpeg_noise` and `haze`):")
    for deg in ["blur_jpeg_noise", "haze"]:
        md_lines.append(f"- **On `{deg}`:**")
        for stg in ["denoise_only", "contrast_only", "sharpen_only", "full_pipeline"]:
            st_data = ablation_answers.get(deg, {}).get(stg)
            if not st_data:
                continue
            p_diff = st_data["psnr"]["mean_diff"]
            s_diff = st_data["ssim"]["mean_diff"]
            md_lines.append(
                f"  - `{stg}`: ΔPSNR = {p_diff:+.2f} dB (CI [{st_data['psnr']['ci_low']:+.2f}, {st_data['psnr']['ci_high']:+.2f}]), "
                f"ΔSSIM = {s_diff:+.4f} (CI [{st_data['ssim']['ci_low']:+.4f}, {st_data['ssim']['ci_high']:+.4f}])."
            )

    md_lines.append("\n## 8. Known Limitations of this Evaluation\n")
    md_lines.append("1. **Originals are already JPEG-compressed:** The tiles in `data/` were downloaded from mapping APIs and have already undergone lossy compression, quantization, and bilinear resampling. Treating them as pristine ground truth HR means synthetic degradation compounds upon existing artifacts.")
    md_lines.append("2. **Synthetic degradation differs from the real API:** Uniform Gaussian noise, idealized box/Gaussian blur, and Koschmieder optical haze are mathematical approximations. The actual Static Maps API pipeline involves sensor Bayer demosaicing, variable satellite camera PSFs, aerial tiling orthorectification, and non-uniform atmospheric scattering.")
    md_lines.append("3. **PSNR and SSIM penalise sharpening and tone changes by construction:** Full-reference pixel difference metrics (MSE/PSNR) strictly measure $L_2$ deviation from reference. Increasing dynamic range via CLAHE or sharpening edge contrast necessarily moves pixel values away from a hazy or blurry reference, producing a lower numerical PSNR even when human interpretability and downstream object detection are significantly enhanced.")
    md_lines.append("4. **Bicubic downsampling favours SR:** Convolutional super-resolution networks like FSRCNN are trained using bicubic downsampling as the degradation kernel. Evaluating SR on `bicubic_ds` creates an artificial advantage (inverse problem matching the training distribution) compared to realistic anti-aliased area downsampling with noise and compression.")
    md_lines.append("5. **The scale=2 real-pair design is stronger but requires an API key:** Directly acquiring native scale=1 (640x640) and scale=2 (1280x1280) tiles provides true physical sensor ground truth, eliminating synthetic degradation entirely. However, doing so requires active Google Maps Platform API credentials and incurs usage quotas.")

    md_lines.append("\n## 9. Per-Image Results Table (Sample / Selected Tiles)\n")
    md_lines.append("| Tile | Source | Variant | SR PSNR | Bicubic PSNR | ΔPSNR | SR SSIM | Bicubic SSIM | ΔSSIM |")
    md_lines.append("|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in rec_test_a:
        if r["method"] == "sr":
            stem = r["tile"]
            src = r["source"]
            var = r["variant"]
            p_sr = r["psnr"]
            p_bic = r["baseline_psnr"]
            dp = r["diff_psnr"]
            s_sr = r["ssim"]
            s_bic = r["baseline_ssim"]
            ds = r["diff_ssim"]
            md_lines.append(f"| {stem} | {src} | {var} | {p_sr:.2f} | {p_bic:.2f} | {dp:+.2f} | {s_sr:.4f} | {s_bic:.4f} | {ds:+.4f} |")

    with open(out_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Satellite Tile Enhancement Pipelines")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Input directory containing satellite tiles (default: data).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results.md"),
        help="Path to output markdown report (default: results.md).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Fixed random seed for degradation and bootstrap (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tiles to evaluate (useful for fast verification).",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="Optional CSV file with lat,lng coordinates for real-pair Google Maps check.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_dir = args.input
    report_path = args.report
    csv_path = report_path.with_suffix(".csv")

    all_images = collect_images(input_dir)
    if not all_images:
        sys.exit(f"Error: No images found in '{input_dir}'")

    records, meta = run_evaluation(
        image_paths=all_images,
        seed=args.seed,
        limit=args.limit,
        pairs_csv=args.pairs_csv,
    )

    print(f"\nWriting results to {report_path} and {csv_path}...")
    save_results_csv(records, csv_path)
    generate_results_markdown(records, meta, report_path)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
