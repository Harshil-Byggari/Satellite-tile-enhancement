# Satellite Tile Enhancement

An end-to-end computer vision and super-resolution pipeline for satellite imagery preprocessing and empirical evaluation, designed for clean CPU execution without GPU or model training dependencies.

---

## 1. Installation

Create a clean virtual environment and install pinned dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Only `opencv-contrib-python`, `numpy`, `scikit-image`, `pillow`, and `tqdm` are used. Ensure standard `opencv-python` is not installed alongside `opencv-contrib-python`.

---

## 2. Models

Pretrained super-resolution weights live in the `models/` directory:
- **Default:** `models/FSRCNN_x2.pb` (Fast Super-Resolution Convolutional Neural Network, scale 2x, ~39 KB).
- **Optional:** `models/EDSR_x2.pb` (Enhanced Deep Residual Networks for Single Image Super-Resolution, scale 2x, ~38.5 MB).

If any model file is missing from disk upon running `enhance.py` or `evaluate.py`, it will **auto-download automatically via `urllib`** from the official upstream repository:
- `https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb`
- `https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb`

---

## 3. CLI Usage & Exact Commands

### A. Classical Enhancement Baseline
Runs Non-Local Means color denoising, luminance level stretching, CLAHE, and unsharp masking with coring threshold.
```bash
python enhance.py --input data/ --output out/classical --method classical
```

### B. Bicubic 2x Upscaling Baseline
Upscales 2x using bicubic interpolation (`cv2.INTER_CUBIC`).
```bash
python enhance.py --input data/ --output out/bicubic --method bicubic
```

### C. Learned Super-Resolution 2x
Upscales 2x using the pretrained FSRCNN model (cached in memory once per run).
```bash
python enhance.py --input data/ --output out/sr --method sr
```
*(Optional: `--sr-model edsr` to run EDSR instead of FSRCNN).*

### D. Comprehensive Evaluation Suite
Evaluates Test A (SR vs Bicubic), Test B (Classical vs degraded baselines and stage ablations), and no-reference metrics on real tiles with 95% bootstrap confidence intervals (2,000 resamples).
```bash
# Fast verification (first 5 tiles):
python evaluate.py --input data/ --report results.md --limit 5

# Full evaluation across all tiles:
python evaluate.py --input data/ --report results.md
```
Generates both `results.md` and `results.csv`.

### D. Comprehensive Evaluation Suite
Evaluates Test A (SR vs Bicubic), Test B (Classical vs degraded baselines and stage ablations), and no-reference metrics on real tiles with 95% bootstrap confidence intervals (2,000 resamples).
```bash
# Fast verification (first 5 tiles):
python evaluate.py --input data/ --report results.md --limit 5

# Full evaluation across all tiles:
python evaluate.py --input data/ --report results.md
```
Generates both `results.md` and `results.csv`.

### E. Visual Inspection Grid
A composite before/after inspection grid is provided at `grid.png` in the repository root, presenting a direct side-by-side comparison across tiles:
1. **Original** (native input resolution)
2. **Classical Enhancement** (1x, denoised, tone-stretched, CLAHE, and sharpened)
3. **Bicubic Upscaling** (2x baseline interpolation)
4. **Learned Super-Resolution** (2x FSRCNN neural network)

---

## 4. Assumptions

1. **Dataset Divergence:** The brief originally described 20 Google Maps 640×640 JPEG tiles. The actual provided dataset consists of **41 PNG tiles** from mixed sources (24 Google and 17 Mapbox tiles). All evaluations are stratified and reported overall as well as independently per source (`google` vs `mapbox`).
2. **Lossless Container Assumption:** While the tiles are stored in PNG format (a lossless container), satellite tiles pulled from Web map APIs have already undergone lossy compression, quantization, and bilinear resampling. The evaluation assumes prior compression artifacts are already embedded in the images.
3. **Reference Generation:** Because the provided tiles represent the highest-resolution version available, synthetic degradation (downsampling, JPEG re-encoding, Gaussian noise, blur, and Koschmieder optical haze) is applied to manufacture ground-truth pairs.
4. **Synthetic Degradation Approximation:** Synthetic degradation kernels (uniform Gaussian noise, spatial Gaussian blur, Koschmieder atmospheric haze) are controlled mathematical models; they approximate, but do not fully replicate, physical sensor optics, atmospheric scattering, and proprietary API server compression pipelines.
5. **Dimensions & Channel Format:** Tiles have mixed dimensions (from 75×103 to 1086×863). To guarantee exact 2x reconstruction shapes during evaluation without distortion, tiles with odd dimensions are cropped by 1 pixel on the boundary. Any alpha channels are normalized to standard 3-channel BGR arrays.
6. **Hardware & Models:** All pipelines run strictly on CPU without GPU requirements. Only published pretrained weights are used (no model training).
7. **Deterministic Execution:** A fixed random seed (default `0`) is used across all stochastic processes (noise injection, bootstrap resampling).
8. **No Mandatory API Keys:** The entire pipeline and evaluation suite run self-contained against `data/`. No Google Maps or Mapbox API keys are required to execute deliverables.
9. **Terms of Service Differences:** Google Maps Platform and Mapbox operate under distinct terms of service regarding tile caching, offline storage, modification, and attribution.
