# Satellite Tile Enhancement: Technical Report

## 1. Pipeline and Ordering Rationale
The classical pipeline sequences operations strictly from low to high frequency:
1. **Denoising first:** Non-Local Means color denoising (`h=3`) runs first so subsequent contrast and edge filters do not amplify sensor noise or JPEG block boundaries.
2. **Luminance stretch and CLAHE:** Processing occurs solely on the $L$ channel in CIELAB. Percentile clipping (1st–99th) cuts haze; CLAHE (`clipLimit=2.0`, `grid=8×8`) enhances local driveway/roof contrast without altering chromatic channels ($a, b$) or causing oversaturation.
3. **Sharpening last with coring:** Gaussian unsharp masking ($\sigma=1.0$, amount$=1.0$) is applied last with a coring threshold of 3.0 to isolate genuine structural boundaries while suppressing flat-field noise.
4. **No white balance:** Omitted because aerial landscapes legitimately feature dominant vegetation greens and soil browns that gray-world balancing destroys.

## 2. Evaluation Design
Ground truth was established by synthetic degradations on the native tiles:
- **What it measures:** Full-reference fidelity (PSNR, SSIM, Sobel gradient Pearson correlation) and no-reference quality (Laplacian variance, wavelet noise $\sigma$, 8×8 blockiness ratio, CIELAB $\Delta E$), tested via 2,000-sample bootstrap 95% confidence intervals.
- **Weaknesses and biases:** Full-reference metrics penalize contrast expansion and sharpening by construction; enhanced dynamic range yields lower PSNR even when visual interpretability improves. Bicubic downsampling (`bicubic_ds`) artificially favors FSRCNN because it matches the model's training kernel. Synthetic Gaussian noise and Koschmieder optical haze ($t=0.7, A=220$) approximate, but simplify, complex real-world sensor optics and non-uniform atmospheric scattering.

## 3. Three Failure Cases
[TODO: I will choose three tiles from data/ and describe them]
- Candidate 1: Edge halo overshoot — `building_4efaf9_mapbox` (halo score: 47.78, unsharp overshoots high-contrast roof boundary into background).
- Candidate 2: Sensor noise amplification — `building_e42da1_google` ($\Delta\sigma = +9.38$, low-light sensor noise amplified by luminance stretching).
- Candidate 3: JPEG blockiness amplification — `building_dc7b9e_mapbox` ($\Delta\text{blockiness} = +0.170$, 8×8 grid boundary contrast heightened by CLAHE).

## 4. 50,000 Tiles/Day Throughput
Measured single-core CPU throughput:
- **Classical:** 0.5416 s/tile $\rightarrow$ ~159,539 tiles/day per core.
- **Learned SR (FSRCNN):** 0.2989 s/tile $\rightarrow$ ~289,085 tiles/day per core.
- **Bicubic:** 0.0026 s/tile $\rightarrow$ ~33,032,573 tiles/day per core.

50,000 tiles/day requires ~7.5 core-hours of classical compute, but Non-Local Means denoising accounts for >90% of latency and scales poorly on large tiles ($>1000\times1000$). At scale, this bottleneck breaks first. Recommended changes: replace NLM with fast guided/bilateral filtering; export FSRCNN to ONNX Runtime with CPU acceleration; distribute across a Celery/Redis worker queue; bypass already clean tiles using a fast no-reference check; and cache identical tile requests.

## 5. Google Maps Platform Terms of Service
[TODO: I will read the current Google Maps Platform Terms of Service myself and fill this in]
Topics to check:
- Caching and permanent database storage limits on static tiles.
- Restrictions on creating derivative works or modifying imagery.
- Attribution, copyright watermark preservation, and logo display requirements.
- Usage outside Google's standard map display environment.
- Distinction between Google Maps Platform and Mapbox Terms of Service.
