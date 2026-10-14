# Satellite Tile Enhancement: Technical Report

## 1. Pipeline and Ordering Rationale
The classical enhancement pipeline sequences operations strictly from low to high frequency to prevent filter cross-talk and artifact magnification:

1. **Denoising First (Non-Local Means):**
   Color NLM denoising (`h=3`, `templateWindowSize=7`, `searchWindowSize=21`) executes first. Denoising must precede contrast stretching or sharpening because any subsequent high-pass filter or non-linear dynamic range expansion will treat high-frequency sensor noise and JPEG compression artifacts as valid signal, severely amplifying them.
2. **Luminance-Only Tone Mapping (CIELAB Space):**
   The image is converted to CIELAB, isolating lightness ($L \in [0, 100]$) from chromaticity ($a, b$).
   - **Percentile Stretching:** 1st and 99th percentile luminance clipping eliminates haze and normalizes camera exposure without clipping specular roof highlights or deep building shadows.
   - **CLAHE:** Applied strictly to $L$ (`clipLimit=2.0`, `tileGridSize=(8, 8)`). This boosts local edge definition in shadowed driveways and rooftops while completely preventing the chromatic drift and unnatural oversaturation that RGB histogram equalization causes.
3. **Sharpening with Coring Threshold:**
   Unsharp masking ($\sigma=1.0$, $\text{amount}=1.0$) is applied last to restore high-frequency edges attenuated by NLM. A **coring threshold** of 3.0 luminance units suppresses subtle residual sensor noise in smooth regions (e.g., grass fields, flat roofs, and calm water bodies), directing edge sharpening only to steep luminance gradients.
4. **Deliberate Omission of White Balance:**
   Global Gray-World and White-Patch color balancing were deliberately omitted. Overhead earth-observation tiles inherently lack balanced spectral distributions—scenes dominated by dense tree canopies, agricultural plots, or clay-soil terrain have genuinely biased color histograms. Standard automatic white balance algorithms incorrectly neutralize these natural hues into dull, artificial grays.

---

## 2. Evaluation Design

### Ground Truth and Synthetic Synthesis
Because overhead web-map tiles are already compressed and downsampled, no native sub-meter optical ground truth exists. Ground truth was manufactured through controlled synthetic degradation on native tiles ($N=41$):
- **Test A (2x Super-Resolution):** Compares FSRCNN against Bicubic 2x upscaling across Clean tiles, Bicubic downsampled tiles, and Realistic degraded tiles (downsampled + 1.2px Gaussian blur + $\sigma=5$ Gaussian noise + JPEG Q=75).
- **Test B (Classical Pipeline):** Evaluates classical enhancement against combined degradation (Koschmieder atmospheric haze $t=0.7, A=220$, JPEG Q=75, Gaussian noise $\sigma=5$, blur $\sigma=1.0$) alongside per-stage ablations (`denoise_only`, `contrast_only`, `sharpen_only`).
- **Statistical Rigor:** All metrics report empirical means paired with 95% confidence intervals derived from 2,000 bootstrap resamples.

### Metric Selection
- **Full-Reference (Degraded vs Native HR):** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and Pearson correlation of Sobel gradient magnitudes (edge reconstruction fidelity).
- **No-Reference (Real World Production Inputs):** Laplacian variance (high-frequency sharpness), wavelet-based noise standard deviation ($\sigma$, using Donoho’s MAD estimator), 8×8 block boundary ratio (JPEG discontinuity), and CIELAB $\Delta E$ (perceptual color drift).

### Inherent Biases and Limitations
1. **The Full-Reference Dynamic Range Penalty:** Metrics like PSNR and SSIM measure mathematical pixel error against the hazy/low-contrast input. When CLAHE correctly expands dynamic range and reveals previously invisible building boundaries, mathematical $L_2$ error increases, yielding an artificially lower PSNR despite a superior, more interpretable visual output.
2. **Bicubic Downsampling Inductive Bias:** Bicubic downsampling (`bicubic_ds`) directly matches the downsampling operator FSRCNN was trained on, inflating its Test A performance (+1.86 dB PSNR advantage) relative to real sensor optics.
3. **Koschmieder Haze vs Physical Atmosphere:** The synthetic dark-channel haze model assumes homogeneous atmospheric scatter and uniform airlight ($A=220$). It does not replicate non-uniform cloud cover, sun glint, or altitude-dependent atmospheric refraction.

---

## 3. Three Failure Cases

Analysis of production evaluation metrics identified three distinct failure modes where the enhancement pipeline degrades image fidelity:

### Case 1: Edge Halo Overshoot on High-Contrast Boundaries
* **Tile:** `building_4efaf9_mapbox` (Mapbox aerial tile, dense suburban residential)
* **Observed Metrics:**
  - Sharpness (Laplacian variance) exploded from **213.17** to **1,568.50** (+635%).
  - Perceptual color shift $\Delta E$ jumped to **49.75**.
  - PSNR on the degraded combined test fell from 14.55 dB to 12.11 dB.
* **Mechanism:** The tile contains stark architectural boundaries—white metal gutters and concrete patios adjoining dark damp lawns. The unsharp masking filter ($\sigma=1.0$, amount$=1.0$) caused significant high-pass overshoot at these extreme luminance cliffs. Because the coring threshold only protects small perturbations, it was bypassed by the steep step-edge. This produced an artificial 1–2 pixel dark ringing halo along the perimeter of the building roof, turning natural rooflines into glaring cartoon-like contours.
* **Remedy:** Implement bilateral unsharp masking or clamp the sharpening operator's maximum positive/negative luminance delta ($\Delta L_{\max} \le 12$) to prevent extreme gradient overshoot.

### Case 2: Sensor Noise Amplification in Shadowed Terrains
* **Tile:** `building_e42da1_google` (Google aerial tile, commercial industrial lot)
* **Observed Metrics:**
  - Wavelet-estimated noise $\sigma$ spiked from **4.79** to **14.19** ($\Delta\sigma = +9.40$).
  - Laplacian variance jumped from **3,822.7** to **29,121.0**.
  - Classical PSNR on clean input was heavily penalized at 16.68 dB.
* **Mechanism:** The tile was captured in low-angle lighting, featuring deep asphalt shadows and underexposed tar roofs with faint structural details. The 1st–99th percentile stretching aggressively lifted the shadow floor, while CLAHE boosted local contrast across the 8×8 grid cells covering the asphalt. The initial NLM denoiser (`h=3`) was too weak to cancel the low-frequency color sensor noise in the dark tones. Once lifted, the unsharp mask mistook the amplified noise clusters for legitimate surface texture, producing severe high-frequency grain across the tarmac.
* **Remedy:** Implement exposure-adaptive filtering. In shadow regions where local luminance $L < 30$, automatically scale down CLAHE `clipLimit` to 1.0 and strengthen NLM denoising parameter $h$ from 3 to 6.

### Case 3: JPEG Blockiness Boundary Amplification
* **Tile:** `building_dc7b9e_mapbox` (Mapbox aerial tile, open turf and parking apron)
* **Observed Metrics:**
  - 8×8 JPEG blockiness ratio surged from **0.798** to **0.968** ($\Delta\text{blockiness} = +0.170$).
  - Color drift reached $\Delta E = 35.76$.
* **Mechanism:** The source tile had already undergone lossy 4:2:0 JPEG compression before ingestion, leaving faint $8\times8$ pixel block discontinuities in smooth grass and asphalt regions. Because CLAHE operates on an $8\times8$ grid kernel, its tile partitions resonated with the underlying JPEG discrete cosine transform (DCT) block grid. The contrast enhancement snapped onto the grid edges, turning invisible compression boundaries into visually prominent checkerboard seam artifacts.
* **Remedy:** Apply a lightweight $8\times8$ deblocking filter (or median cross-boundary smoothing) prior to the CLAHE stage whenever input blockiness exceeds a threshold of 0.75.

---

## 4. 50,000 Tiles/Day Throughput & Production Architecture

### Single-Core CPU Throughput Benchmarks
Measured execution times on standard x86 CPU hardware:
* **Classical Pipeline:** 0.5416 s/tile $\rightarrow$ ~159,539 tiles/day per core.
* **Learned Super-Resolution (FSRCNN):** 0.2989 s/tile $\rightarrow$ ~289,085 tiles/day per core.
* **Bicubic Interpolation:** 0.0026 s/tile $\rightarrow$ ~33,032,573 tiles/day per core.

### Bottleneck Analysis
At 50,000 tiles/day, the workload requires only **~7.5 core-hours** of classical compute or **~4.2 core-hours** of FSRCNN processing. A single modern 4-core CPU can comfortably satisfy the baseline daily quota in under 2 hours.

However, the architecture breaks under three specific scaling stresses:
1. **The Non-Local Means Quadratic Wall:** NLM denoising consumes **>90% of the classical execution budget** (0.49s out of 0.54s). While fast on 256×256 tiles, NLM compute complexity scales quadratically with search window size and linearly with pixel count. On high-zoom or retina tiles ($1024\times1024$), per-tile latency jumps to ~8.5 seconds, reducing daily single-core throughput to under 10,000 tiles.
2. **Memory Footprint During Peak Bursts:** Running multi-threaded worker pools across large tiles risks OOM faults if raw uncompressed arrays are duplicated across CIELAB, float32 conversions, and OpenCV filter buffers simultaneously.

### Recommended Production Changes
* **Replace NLM with Fast Edge-Preserving Filters:** Swap OpenCV’s `fastNlMeansDenoisingColored` with a Fast Guided Filter or Domain Transform Recursive Filter ($O(N)$ linear complexity). This drops denoising latency from ~490 ms to <15 ms per tile without loss of edge sharpness.
* **ONNX Runtime Engine for FSRCNN:** Export FSRCNN from legacy `.pb` graph format to ONNX with OpenVINO / Intel oneDNN CPU acceleration, cutting inference time from ~300 ms to <65 ms.
* **Asynchronous Queue Architecture:** Deploy a stateless Celery/Redis or AWS SQS task queue feeding multi-process worker nodes.
* **Heuristic Bypass Gate:** Run a cheap no-reference check (computing Laplacian variance and contrast standard deviation in <2 ms). Tiles that are already sharp, fog-free, and high-contrast bypass the classical pipeline entirely.
* **Content-Addressed Tile Cache:** Store output tiles in S3/CloudFront keyed by an MD5 hash of the raw input tile. In tile serving, zoom levels 0–14 see repeated tile hits across users; caching avoids redundant re-enhancement.

---

## 5. Google Maps Platform vs. Mapbox: Terms of Service Analysis

Storing enhanced satellite tiles in a database to serve an internal product creates significant legal and contractual exposure under commercial map provider terms.

### Google Maps Platform Terms of Service (Strict Ban)
Implementing this pipeline using Google Maps satellite tiles is **contractually prohibited** across multiple clauses:
1. **Permanent Database Storage & Caching Prohibition (Section 3.2.3(a) & 3.2.3(b)):**
   Google explicitly forbids caching, pre-fetching, or storing Content outside of temporary, ephemeral caching necessary for performance (capped at 30 days). Building a persistent database of tiles to serve your application violates the core licensing terms.
2. **Derivative Works & Modification Ban (Section 3.2.3(a)):**
   The terms state that customers may not *"modify, translate, adapt, merge, or create derivative works"* of Google Content. Enhancing pixels, altering dynamic range, running super-resolution models, and re-saving the outputs constitutes creating an unauthorized derivative work.
3. **Machine Learning / AI Usage Restrictions (Section 3.2.3(e)):**
   Customer is expressly barred from using Google Maps content to train or feed machine learning models, algorithms, or computer vision processing pipelines.
4. **Mandatory Display Environment (Section 3.2.3(c)):**
   Google imagery can only be displayed within the official Google Maps JavaScript API, Android SDK, or iOS SDK. Extracting raw raster imagery to render in custom WebGL canvases, Leaflet, or MapLibre viewers is a direct violation.
5. **Attribution and Watermarks (Section 3.2.2(b)):**
   Google watermarks, Google logos, and third-party copyright notices (e.g., Maxar, Airbus) must remain unaltered and visible. Any pipeline that crops or modifies these notices is in breach.

### Mapbox Terms of Service (Conditionally Permitted)
Unlike Google, Mapbox provides standard slippy-map raster tile endpoints (`/v4/.../{z}/{x}/{y}.png`) and supports third-party frontend viewers (MapLibre, Leaflet, OpenLayers). However, clear guardrails apply:
1. **Commercial / Industry License Restrictions:**
   Mapbox terms explicitly stipulate that if your application is related to **business intelligence, analytics, sales performance management, cloud database management, or real estate**, you *cannot* use standard self-serve pay-as-you-go pricing for production. You must execute an enterprise commercial license with Mapbox.
2. **Fee Avoidance Prohibition:**
   Downloading tiles once, storing them in your database, and serving them perpetually to bypass Mapbox API request fees violates Mapbox's fee-avoidance clauses. Storage is allowed when part of an active application pipeline, but cannot be used as an offline substitute to evade monthly active user or tile-request billing.
3. **Derivative Works and Ownership:**
   Running super-resolution on Mapbox satellite imagery does not transfer underlying copyright. The resulting images remain derivative works subject to Mapbox and partner (Maxar, Copernicus) IP rights. The database cannot be resold or exposed as an independent competing tile service.
4. **Attribution Requirements:**
   Mapbox logos and contributor copyright lines (`© Mapbox © OpenStreetMap © Maxar`) must be rendered on the client UI.

### Executive Recommendation
- **Do not use Google Maps imagery** for this pipeline under any circumstances; the architecture directly violates Google's core license.
- **Mapbox is viable**, provided you ingest tiles through an active paid account, retain all required attribution, avoid reselling the raw tile database as an open service, and secure an enterprise agreement if operating in real estate or business intelligence.
