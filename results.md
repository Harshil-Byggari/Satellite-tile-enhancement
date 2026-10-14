# Satellite Tile Enhancement Evaluation Report

## 1. Assumptions and Setup

- **Random Seed:** `0` (fixed for all stochastic operations: noise, resampling, bootstrap).
- **Evaluated Tiles:** 41 tiles from `data/` (Google and Mapbox mixed sources).
- **Hardware Context:** CPU execution only (Intel/AMD x86_64, no GPU, no model training).
- **Super-Resolution Model:** FSRCNN 2x (`models/FSRCNN_x2.pb`, cached in memory).
- **Synthetic Degradation Parameters:**
  - **Downsampling:** 2x factor via OpenCV `INTER_AREA` (realistic/clean) and `INTER_CUBIC` (bicubic downsampled).
  - **JPEG Compression:** In-memory encode/decode at quality 60 (Test B blur/noise), 65 (combined), 70 (Test A realistic).
  - **Gaussian Blur:** $\sigma = 1.2$ (Test B), $\sigma = 1.0$ (combined).
  - **Gaussian Noise:** $\sigma = 3.0$ (zero-mean).
  - **Haze Model:** $I' = I \cdot t + A \cdot (1 - t)$ with transmission $t = 0.7$, airlight $A = 220.0$.
- **Statistical Rigor:** 95% two-sided empirical bootstrap confidence interval (2000 resamples, seeded). CI containing zero indicates *no clear difference*.

## 2. Test A: Super-Resolution (FSRCNN) vs Bicubic Baseline (2x)

Each LR tile is upscaled 2x using FSRCNN and Bicubic interpolation, then evaluated against the original HR reference.

### Aggregate Performance (Mean ± Std)

| Variant | Source | Method | PSNR (dB) | SSIM | Sobel Edge Corr |
|:---|:---|:---|:---:|:---:|:---:|
| clean | Overall | BICUBIC | 32.53 ± 6.36 | 0.9127 ± 0.0915 | 0.9173 ± 0.0414 |
| clean | Overall | SR | 33.74 ± 6.66 | 0.9186 ± 0.0888 | 0.9493 ± 0.0372 |
| clean | Google | BICUBIC | 27.54 ± 2.40 | 0.8577 ± 0.0837 | 0.8934 ± 0.0374 |
| clean | Google | SR | 28.51 ± 2.66 | 0.8683 ± 0.0858 | 0.9269 ± 0.0339 |
| clean | Mapbox | BICUBIC | 39.58 ± 2.14 | 0.9903 ± 0.0038 | 0.9510 ± 0.0147 |
| clean | Mapbox | SR | 41.13 ± 1.96 | 0.9897 ± 0.0025 | 0.9808 ± 0.0044 |
| bicubic_ds | Overall | BICUBIC | 32.53 ± 6.36 | 0.9163 ± 0.0881 | 0.9191 ± 0.0442 |
| bicubic_ds | Overall | SR | 34.06 ± 6.84 | 0.9235 ± 0.0866 | 0.9498 ± 0.0401 |
| bicubic_ds | Google | BICUBIC | 27.55 ± 2.46 | 0.8643 ± 0.0820 | 0.8947 ± 0.0414 |
| bicubic_ds | Google | SR | 28.69 ± 2.80 | 0.8767 ± 0.0867 | 0.9262 ± 0.0373 |
| bicubic_ds | Mapbox | BICUBIC | 39.55 ± 2.16 | 0.9897 ± 0.0041 | 0.9536 ± 0.0165 |
| bicubic_ds | Mapbox | SR | 41.63 ± 1.93 | 0.9897 ± 0.0026 | 0.9830 ± 0.0050 |
| realistic | Overall | BICUBIC | 29.86 ± 4.74 | 0.8061 ± 0.0929 | 0.8714 ± 0.0425 |
| realistic | Overall | SR | 30.54 ± 4.88 | 0.8242 ± 0.0944 | 0.8954 ± 0.0416 |
| realistic | Google | BICUBIC | 26.12 ± 1.92 | 0.7441 ± 0.0739 | 0.8489 ± 0.0409 |
| realistic | Google | SR | 26.69 ± 2.01 | 0.7619 ± 0.0763 | 0.8748 ± 0.0408 |
| realistic | Mapbox | BICUBIC | 35.14 ± 1.15 | 0.8935 ± 0.0067 | 0.9031 ± 0.0169 |
| realistic | Mapbox | SR | 35.97 ± 1.10 | 0.9122 ± 0.0061 | 0.9245 ± 0.0193 |

### Paired Difference Analysis (SR minus Bicubic)

| Variant | Source | Metric | Mean Diff | 95% Bootstrap CI | Win Rate (%) | Conclusion |
|:---|:---|:---|:---:|:---:|:---:|:---|
| clean | Overall | PSNR (dB) | +1.2135 | [+1.0275, +1.3997] | 100.0% | statistically superior (+) |
| clean | Overall | SSIM | +0.0059 | [+0.0033, +0.0087] | 65.9% | statistically superior (+) |
| clean | Overall | Sobel Corr | +0.0320 | [+0.0278, +0.0365] | 100.0% | statistically superior (+) |
| clean | Google | PSNR (dB) | +0.9720 | [+0.7423, +1.2100] | 100.0% | statistically superior (+) |
| clean | Google | SSIM | +0.0106 | [+0.0072, +0.0140] | 95.8% | statistically superior (+) |
| clean | Google | Sobel Corr | +0.0336 | [+0.0279, +0.0400] | 100.0% | statistically superior (+) |
| clean | Mapbox | PSNR (dB) | +1.5543 | [+1.3263, +1.7776] | 100.0% | statistically superior (+) |
| clean | Mapbox | SSIM | -0.0006 | [-0.0014, +0.0002] | 23.5% | no clear difference |
| clean | Mapbox | Sobel Corr | +0.0297 | [+0.0241, +0.0356] | 100.0% | statistically superior (+) |
| bicubic_ds | Overall | PSNR (dB) | +1.5269 | [+1.2918, +1.7541] | 100.0% | statistically superior (+) |
| bicubic_ds | Overall | SSIM | +0.0072 | [+0.0039, +0.0107] | 73.2% | statistically superior (+) |
| bicubic_ds | Overall | Sobel Corr | +0.0306 | [+0.0260, +0.0353] | 100.0% | statistically superior (+) |
| bicubic_ds | Google | PSNR (dB) | +1.1350 | [+0.8490, +1.4276] | 100.0% | statistically superior (+) |
| bicubic_ds | Google | SSIM | +0.0123 | [+0.0074, +0.0171] | 95.8% | statistically superior (+) |
| bicubic_ds | Google | Sobel Corr | +0.0315 | [+0.0252, +0.0382] | 100.0% | statistically superior (+) |
| bicubic_ds | Mapbox | PSNR (dB) | +2.0801 | [+1.8942, +2.2595] | 100.0% | statistically superior (+) |
| bicubic_ds | Mapbox | SSIM | -0.0001 | [-0.0009, +0.0008] | 41.2% | no clear difference |
| bicubic_ds | Mapbox | Sobel Corr | +0.0294 | [+0.0235, +0.0356] | 100.0% | statistically superior (+) |
| realistic | Overall | PSNR (dB) | +0.6778 | [+0.5775, +0.7795] | 100.0% | statistically superior (+) |
| realistic | Overall | SSIM | +0.0182 | [+0.0168, +0.0197] | 100.0% | statistically superior (+) |
| realistic | Overall | Sobel Corr | +0.0240 | [+0.0193, +0.0289] | 100.0% | statistically superior (+) |
| realistic | Google | PSNR (dB) | +0.5707 | [+0.4391, +0.7167] | 100.0% | statistically superior (+) |
| realistic | Google | SSIM | +0.0178 | [+0.0155, +0.0204] | 100.0% | statistically superior (+) |
| realistic | Google | Sobel Corr | +0.0259 | [+0.0197, +0.0327] | 100.0% | statistically superior (+) |
| realistic | Mapbox | PSNR (dB) | +0.8291 | [+0.7092, +0.9444] | 100.0% | statistically superior (+) |
| realistic | Mapbox | SSIM | +0.0187 | [+0.0180, +0.0194] | 100.0% | statistically superior (+) |
| realistic | Mapbox | Sobel Corr | +0.0214 | [+0.0151, +0.0279] | 100.0% | statistically superior (+) |

## 3. Test B: Classical Enhancement Pipeline Evaluation

Evaluation of degraded baseline vs classical enhanced pipeline (`enhance_classical`) scored against ground truth HR.

### Aggregate Performance (Mean ± Std)

| Degradation | Source | Pipeline | PSNR (dB) | SSIM | Sobel Edge Corr |
|:---|:---|:---|:---:|:---:|:---:|
| blur_jpeg_noise | Overall | degraded_baseline | 29.21 ± 4.71 | 0.7791 ± 0.0967 | 0.8546 ± 0.0490 |
| blur_jpeg_noise | Overall | classical_enhanced | 17.94 ± 2.86 | 0.6076 ± 0.0980 | 0.8221 ± 0.0513 |
| blur_jpeg_noise | Google | degraded_baseline | 25.49 ± 1.92 | 0.7170 ± 0.0816 | 0.8271 ± 0.0458 |
| blur_jpeg_noise | Google | classical_enhanced | 17.65 ± 2.93 | 0.5601 ± 0.0871 | 0.7986 ± 0.0524 |
| blur_jpeg_noise | Mapbox | degraded_baseline | 34.47 ± 1.11 | 0.8668 ± 0.0059 | 0.8936 ± 0.0154 |
| blur_jpeg_noise | Mapbox | classical_enhanced | 18.35 ± 2.69 | 0.6746 ± 0.0693 | 0.8554 ± 0.0241 |
| haze | Overall | degraded_baseline | 15.19 ± 1.09 | 0.7925 ± 0.0459 | 0.9996 ± 0.0004 |
| haze | Overall | classical_enhanced | 18.61 ± 2.58 | 0.6388 ± 0.1239 | 0.9559 ± 0.0202 |
| haze | Google | degraded_baseline | 15.40 ± 1.18 | 0.8003 ± 0.0434 | 0.9999 ± 0.0000 |
| haze | Google | classical_enhanced | 18.69 ± 2.32 | 0.6305 ± 0.1266 | 0.9478 ± 0.0203 |
| haze | Mapbox | degraded_baseline | 14.89 ± 0.87 | 0.7815 ± 0.0471 | 0.9993 ± 0.0003 |
| haze | Mapbox | classical_enhanced | 18.51 ± 2.90 | 0.6506 ± 0.1190 | 0.9673 ± 0.0134 |
| combined | Overall | degraded_baseline | 14.99 ± 0.96 | 0.6097 ± 0.0862 | 0.8739 ± 0.0397 |
| combined | Overall | classical_enhanced | 18.72 ± 3.09 | 0.6903 ± 0.0965 | 0.8681 ± 0.0564 |
| combined | Google | degraded_baseline | 15.05 ± 1.02 | 0.5604 ± 0.0748 | 0.8569 ± 0.0432 |
| combined | Google | classical_enhanced | 18.51 ± 3.08 | 0.6348 ± 0.0853 | 0.8445 ± 0.0621 |
| combined | Mapbox | degraded_baseline | 14.91 ± 0.85 | 0.6793 ± 0.0419 | 0.8979 ± 0.0137 |
| combined | Mapbox | classical_enhanced | 19.02 ± 3.07 | 0.7686 ± 0.0415 | 0.9013 ± 0.0184 |
| clean_null | Overall | degraded_baseline | Inf (exact) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| clean_null | Overall | classical_enhanced | 17.94 ± 2.42 | 0.6082 ± 0.1149 | 0.9456 ± 0.0275 |
| clean_null | Google | degraded_baseline | Inf (exact) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| clean_null | Google | classical_enhanced | 17.81 ± 2.27 | 0.5928 ± 0.1175 | 0.9323 ± 0.0267 |
| clean_null | Mapbox | degraded_baseline | Inf (exact) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| clean_null | Mapbox | classical_enhanced | 18.13 ± 2.60 | 0.6300 ± 0.1074 | 0.9644 ± 0.0145 |

### Paired Difference Analysis (Enhanced minus Baseline)

| Degradation | Source | Metric | Mean Diff | 95% Bootstrap CI | Win Rate (%) | Conclusion |
|:---|:---|:---|:---:|:---:|:---:|:---|
| blur_jpeg_noise | Overall | PSNR (dB) | -11.2719 | [-12.8057, -9.7664] | 0.0% | statistically inferior (-) |
| blur_jpeg_noise | Overall | SSIM | -0.1716 | [-0.1930, -0.1490] | 0.0% | statistically inferior (-) |
| blur_jpeg_noise | Overall | Sobel Corr | -0.0325 | [-0.0401, -0.0255] | 2.4% | statistically inferior (-) |
| blur_jpeg_noise | Google | PSNR (dB) | -7.8424 | [-8.9950, -6.7115] | 0.0% | statistically inferior (-) |
| blur_jpeg_noise | Google | SSIM | -0.1569 | [-0.1834, -0.1291] | 0.0% | statistically inferior (-) |
| blur_jpeg_noise | Google | Sobel Corr | -0.0285 | [-0.0404, -0.0193] | 4.2% | statistically inferior (-) |
| blur_jpeg_noise | Mapbox | PSNR (dB) | -16.1136 | [-17.3642, -14.9925] | 0.0% | statistically inferior (-) |
| blur_jpeg_noise | Mapbox | SSIM | -0.1923 | [-0.2230, -0.1598] | 0.0% | statistically inferior (-) |
| blur_jpeg_noise | Mapbox | Sobel Corr | -0.0382 | [-0.0475, -0.0294] | 0.0% | statistically inferior (-) |
| haze | Overall | PSNR (dB) | +3.4245 | [+2.6117, +4.2593] | 92.7% | statistically superior (+) |
| haze | Overall | SSIM | -0.1536 | [-0.1878, -0.1177] | 17.1% | statistically inferior (-) |
| haze | Overall | Sobel Corr | -0.0438 | [-0.0503, -0.0376] | 0.0% | statistically inferior (-) |
| haze | Google | PSNR (dB) | +3.2873 | [+2.2198, +4.2829] | 95.8% | statistically superior (+) |
| haze | Google | SSIM | -0.1698 | [-0.2122, -0.1218] | 16.7% | statistically inferior (-) |
| haze | Google | Sobel Corr | -0.0521 | [-0.0602, -0.0444] | 0.0% | statistically inferior (-) |
| haze | Mapbox | PSNR (dB) | +3.6182 | [+2.2054, +4.8933] | 88.2% | statistically superior (+) |
| haze | Mapbox | SSIM | -0.1309 | [-0.1794, -0.0762] | 17.6% | statistically inferior (-) |
| haze | Mapbox | Sobel Corr | -0.0320 | [-0.0388, -0.0261] | 0.0% | statistically inferior (-) |
| combined | Overall | PSNR (dB) | +3.7315 | [+2.7683, +4.7100] | 85.4% | statistically superior (+) |
| combined | Overall | SSIM | +0.0806 | [+0.0647, +0.0960] | 97.6% | statistically superior (+) |
| combined | Overall | Sobel Corr | -0.0059 | [-0.0174, +0.0020] | 63.4% | no clear difference |
| combined | Google | PSNR (dB) | +3.4633 | [+2.1580, +4.6233] | 83.3% | statistically superior (+) |
| combined | Google | SSIM | +0.0744 | [+0.0479, +0.0963] | 95.8% | statistically superior (+) |
| combined | Google | Sobel Corr | -0.0124 | [-0.0310, -0.0003] | 54.2% | statistically inferior (-) |
| combined | Mapbox | PSNR (dB) | +4.1101 | [+2.6356, +5.4424] | 88.2% | statistically superior (+) |
| combined | Mapbox | SSIM | +0.0894 | [+0.0718, +0.1098] | 100.0% | statistically superior (+) |
| combined | Mapbox | Sobel Corr | +0.0033 | [-0.0026, +0.0091] | 76.5% | no clear difference |
| clean_null | Overall | PSNR (dB) | -82.0580 | [-82.7903, -81.2845] | 0.0% | statistically inferior (-) |
| clean_null | Overall | SSIM | -0.3918 | [-0.4265, -0.3555] | 0.0% | statistically inferior (-) |
| clean_null | Overall | Sobel Corr | -0.0544 | [-0.0630, -0.0463] | 0.0% | statistically inferior (-) |
| clean_null | Google | PSNR (dB) | -82.1930 | [-83.1204, -81.3345] | 0.0% | statistically inferior (-) |
| clean_null | Google | SSIM | -0.4072 | [-0.4506, -0.3594] | 0.0% | statistically inferior (-) |
| clean_null | Google | Sobel Corr | -0.0677 | [-0.0787, -0.0576] | 0.0% | statistically inferior (-) |
| clean_null | Mapbox | PSNR (dB) | -81.8674 | [-83.1852, -80.6658] | 0.0% | statistically inferior (-) |
| clean_null | Mapbox | SSIM | -0.3700 | [-0.4194, -0.3192] | 0.0% | statistically inferior (-) |
| clean_null | Mapbox | Sobel Corr | -0.0356 | [-0.0426, -0.0290] | 0.0% | statistically inferior (-) |

### Per-Stage Classical Pipeline Ablation Study

Ablation of individual components on `blur_jpeg_noise` and `haze` degradations (Overall, n=41):

| Degradation | Stage Component | Mean ΔPSNR (dB) | 95% CI ΔPSNR | Mean ΔSSIM | 95% CI ΔSSIM | Mean ΔSobel | 95% CI ΔSobel |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| blur_jpeg_noise | denoise_only | +0.881 | [+0.576, +1.211] | +0.0687 | [+0.0572, +0.0801] | -0.0003 | [-0.0036, +0.0030] |
| blur_jpeg_noise | contrast_only | -10.262 | [-11.831, -8.755] | -0.2163 | [-0.2418, -0.1903] | -0.0774 | [-0.0908, -0.0653] |
| blur_jpeg_noise | sharpen_only | -0.975 | [-1.428, -0.557] | -0.1080 | [-0.1232, -0.0930] | +0.0095 | [+0.0039, +0.0146] |
| blur_jpeg_noise | full_pipeline | -11.272 | [-12.806, -9.766] | -0.1716 | [-0.1930, -0.1490] | -0.0325 | [-0.0401, -0.0255] |
| haze | denoise_only | -0.005 | [-0.020, +0.005] | -0.0451 | [-0.0567, -0.0339] | -0.0106 | [-0.0126, -0.0088] |
| haze | contrast_only | +4.544 | [+3.683, +5.423] | -0.1124 | [-0.1483, -0.0730] | -0.0224 | [-0.0268, -0.0187] |
| haze | sharpen_only | +0.040 | [+0.028, +0.054] | +0.0053 | [+0.0036, +0.0070] | -0.0193 | [-0.0202, -0.0184] |
| haze | full_pipeline | +3.425 | [+2.612, +4.259] | -0.1536 | [-0.1878, -0.1177] | -0.0438 | [-0.0503, -0.0376] |

## 4. No-Reference Evaluation on Real Satellite Tiles

Quantification of real image properties without ground truth:

- **Laplacian Variance:** Perceived sharpness / edge energy (higher = sharper).
- **Noise Sigma ($\sigma$):** Robust wavelet-estimated Gaussian noise std (lower = cleaner).
- **Blockiness Ratio:** Step across 8x8 JPEG grid vs interior (1.0 = no grid artifacts, >1.0 = blocking).
- **Contrast:** Standard deviation of CIELAB L channel (higher = broader dynamic range).
- **Colour Shift ($\Delta E$):** Mean Euclidean distance in CIELAB space.

### Real Input vs Classical Enhanced (1x Resolution)

| Source | Method | Sharpness (Lap Var) | Noise ($\sigma$) | Blockiness Ratio | Contrast (Std L) | Colour Shift ($\Delta E$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Overall | Input | 1594.1 ± 1839.4 | 0.60 ± 1.11 | 0.980 ± 0.098 | 58.77 ± 13.88 | 0.00 (Ref) |
| Overall | Classical | 9924.9 ± 12373.6 | 1.70 ± 3.33 | 0.997 ± 0.047 | 84.35 ± 11.82 | 27.92 ± 7.72 |
| Google | Input | 2604.1 ± 1820.0 | 1.01 ± 1.31 | 0.999 ± 0.063 | 61.95 ± 15.10 | 0.00 (Ref) |
| Google | Classical | 16397.9 ± 12662.9 | 2.81 ± 4.00 | 1.002 ± 0.045 | 85.09 ± 11.61 | 27.98 ± 7.58 |
| Mapbox | Input | 168.1 ± 99.5 | 0.03 ± 0.06 | 0.954 ± 0.127 | 54.28 ± 10.39 | 0.00 (Ref) |
| Mapbox | Classical | 786.5 ± 469.1 | 0.13 ± 0.20 | 0.991 ± 0.049 | 83.31 ± 12.04 | 27.83 ± 7.91 |

### Bicubic-2x vs SR-2x (2x Resolution)

| Source | Method | Sharpness (Lap Var) | Noise ($\sigma$) | Blockiness Ratio | Contrast (Std L) | Colour Shift ($\Delta E$ vs Bicubic) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Overall | BICUBIC_2X | 133.9 ± 147.1 | 0.12 ± 0.17 | 1.017 ± 0.126 | 58.65 ± 13.87 | 0.00 (Ref) |
| Overall | SR_2X | 201.1 ± 215.7 | 0.22 ± 0.22 | 1.066 ± 0.098 | 58.76 ± 13.92 | 1.21 ± 0.64 |
| Google | BICUBIC_2X | 217.9 ± 141.0 | 0.19 ± 0.18 | 1.072 ± 0.105 | 61.75 ± 15.13 | 0.00 (Ref) |
| Google | SR_2X | 324.2 ± 206.8 | 0.31 ± 0.22 | 1.098 ± 0.097 | 61.84 ± 15.20 | 1.66 ± 0.46 |
| Mapbox | BICUBIC_2X | 15.3 ± 8.6 | 0.01 ± 0.03 | 0.939 ± 0.112 | 54.27 ± 10.39 | 0.00 (Ref) |
| Mapbox | SR_2X | 27.3 ± 14.3 | 0.09 ± 0.13 | 1.020 ± 0.079 | 54.40 ± 10.44 | 0.59 ± 0.08 |

## 5. Runtime and Throughput

| Method | Total Time (s) | Avg Seconds / Tile | Single-Core Tiles / Day |
|:---|:---:|:---:|:---:|
| Classical Enhancement | 12.21 s | 0.2978 s | ~290,081 |
| Bicubic 2x Upscaling | 0.06 s | 0.0015 s | ~59,520,229 |
| Learned SR (FSRCNN 2x) | 8.03 s | 0.1959 s | ~441,055 |

## 6. Real-Pair Check (Google Static Maps scale=1 vs scale=2)

Real-pair check: skipped (not configured).


## 7. Answers to Core Evaluation Questions

### Question 1: Does Learned Super-Resolution (FSRCNN) beat Bicubic Upscaling?

#### Variant: `clean`
- **Overall:** **YES, SR beats Bicubic** on overall (Mean ΔPSNR = +1.21 dB, 95% CI [+1.028, +1.400], Win Rate = 100.0%).
- **Google:** **YES, SR beats Bicubic** on google (Mean ΔPSNR = +0.97 dB, 95% CI [+0.742, +1.210], Win Rate = 100.0%).
- **Mapbox:** **YES, SR beats Bicubic** on mapbox (Mean ΔPSNR = +1.55 dB, 95% CI [+1.326, +1.778], Win Rate = 100.0%).

#### Variant: `bicubic_ds`
- **Overall:** **YES, SR beats Bicubic** on overall (Mean ΔPSNR = +1.53 dB, 95% CI [+1.292, +1.754], Win Rate = 100.0%).
- **Google:** **YES, SR beats Bicubic** on google (Mean ΔPSNR = +1.14 dB, 95% CI [+0.849, +1.428], Win Rate = 100.0%).
- **Mapbox:** **YES, SR beats Bicubic** on mapbox (Mean ΔPSNR = +2.08 dB, 95% CI [+1.894, +2.260], Win Rate = 100.0%).

#### Variant: `realistic`
- **Overall:** **YES, SR beats Bicubic** on overall (Mean ΔPSNR = +0.68 dB, 95% CI [+0.577, +0.780], Win Rate = 100.0%).
- **Google:** **YES, SR beats Bicubic** on google (Mean ΔPSNR = +0.57 dB, 95% CI [+0.439, +0.717], Win Rate = 100.0%).
- **Mapbox:** **YES, SR beats Bicubic** on mapbox (Mean ΔPSNR = +0.83 dB, 95% CI [+0.709, +0.944], Win Rate = 100.0%).

### Question 2: Does the Task 1 Classical Pipeline Help or Hurt?

#### Degradation: `blur_jpeg_noise`
- **Overall:** **HURTS** on overall across both PSNR (-11.27 dB, CI [-12.806, -9.766]) and SSIM (-0.1716, CI [-0.1930, -0.1490]).
- **Google:** **HURTS** on google across both PSNR (-7.84 dB, CI [-8.995, -6.711]) and SSIM (-0.1569, CI [-0.1834, -0.1291]).
- **Mapbox:** **HURTS** on mapbox across both PSNR (-16.11 dB, CI [-17.364, -14.993]) and SSIM (-0.1923, CI [-0.2230, -0.1598]).

#### Degradation: `haze`
- **Overall:** **Mixed or No clear difference** on overall (ΔPSNR CI [+2.612, +4.259], ΔSSIM CI [-0.1878, -0.1177]).
- **Google:** **Mixed or No clear difference** on google (ΔPSNR CI [+2.220, +4.283], ΔSSIM CI [-0.2122, -0.1218]).
- **Mapbox:** **Mixed or No clear difference** on mapbox (ΔPSNR CI [+2.205, +4.893], ΔSSIM CI [-0.1794, -0.0762]).

#### Degradation: `combined`
- **Overall:** **HELPS** on overall across both PSNR (+3.73 dB, CI [+2.768, +4.710]) and SSIM (+0.0806, CI [+0.0647, +0.0960]).
- **Google:** **HELPS** on google across both PSNR (+3.46 dB, CI [+2.158, +4.623]) and SSIM (+0.0744, CI [+0.0479, +0.0963]).
- **Mapbox:** **HELPS** on mapbox across both PSNR (+4.11 dB, CI [+2.636, +5.442]) and SSIM (+0.0894, CI [+0.0718, +0.1098]).

#### Degradation: `clean_null`
- **Overall:** **HURTS** on overall across both PSNR (-82.06 dB, CI [-82.790, -81.285]) and SSIM (-0.3918, CI [-0.4265, -0.3555]).
- **Google:** **HURTS** on google across both PSNR (-82.19 dB, CI [-83.120, -81.335]) and SSIM (-0.4072, CI [-0.4506, -0.3594]).
- **Mapbox:** **HURTS** on mapbox across both PSNR (-81.87 dB, CI [-83.185, -80.666]) and SSIM (-0.3700, CI [-0.4194, -0.3192]).

#### Per-Stage Ablation Findings (`blur_jpeg_noise` and `haze`):
- **On `blur_jpeg_noise`:**
  - `denoise_only`: ΔPSNR = +0.88 dB (CI [+0.58, +1.21]), ΔSSIM = +0.0687 (CI [+0.0572, +0.0801]).
  - `contrast_only`: ΔPSNR = -10.26 dB (CI [-11.83, -8.75]), ΔSSIM = -0.2163 (CI [-0.2418, -0.1903]).
  - `sharpen_only`: ΔPSNR = -0.98 dB (CI [-1.43, -0.56]), ΔSSIM = -0.1080 (CI [-0.1232, -0.0930]).
  - `full_pipeline`: ΔPSNR = -11.27 dB (CI [-12.81, -9.77]), ΔSSIM = -0.1716 (CI [-0.1930, -0.1490]).
- **On `haze`:**
  - `denoise_only`: ΔPSNR = -0.00 dB (CI [-0.02, +0.01]), ΔSSIM = -0.0451 (CI [-0.0567, -0.0339]).
  - `contrast_only`: ΔPSNR = +4.54 dB (CI [+3.68, +5.42]), ΔSSIM = -0.1124 (CI [-0.1483, -0.0730]).
  - `sharpen_only`: ΔPSNR = +0.04 dB (CI [+0.03, +0.05]), ΔSSIM = +0.0053 (CI [+0.0036, +0.0070]).
  - `full_pipeline`: ΔPSNR = +3.42 dB (CI [+2.61, +4.26]), ΔSSIM = -0.1536 (CI [-0.1878, -0.1177]).

## 8. Known Limitations of this Evaluation

1. **Originals are already JPEG-compressed:** The tiles in `data/` were downloaded from mapping APIs and have already undergone lossy compression, quantization, and bilinear resampling. Treating them as pristine ground truth HR means synthetic degradation compounds upon existing artifacts.
2. **Synthetic degradation differs from the real API:** Uniform Gaussian noise, idealized box/Gaussian blur, and Koschmieder optical haze are mathematical approximations. The actual Static Maps API pipeline involves sensor Bayer demosaicing, variable satellite camera PSFs, aerial tiling orthorectification, and non-uniform atmospheric scattering.
3. **PSNR and SSIM penalise sharpening and tone changes by construction:** Full-reference pixel difference metrics (MSE/PSNR) strictly measure $L_2$ deviation from reference. Increasing dynamic range via CLAHE or sharpening edge contrast necessarily moves pixel values away from a hazy or blurry reference, producing a lower numerical PSNR even when human interpretability and downstream object detection are significantly enhanced.
4. **Bicubic downsampling favours SR:** Convolutional super-resolution networks like FSRCNN are trained using bicubic downsampling as the degradation kernel. Evaluating SR on `bicubic_ds` creates an artificial advantage (inverse problem matching the training distribution) compared to realistic anti-aliased area downsampling with noise and compression.
5. **The scale=2 real-pair design is stronger but requires an API key:** Directly acquiring native scale=1 (640x640) and scale=2 (1280x1280) tiles provides true physical sensor ground truth, eliminating synthetic degradation entirely. However, doing so requires active Google Maps Platform API credentials and incurs usage quotas.

## 9. Per-Image Results Table (Test A & Full Metrics Pointer)

> **Note on Full Per-Image Records:** Comprehensive per-tile records for Test A, Test B (including all baseline variants and stage ablations), and no-reference evaluation on all 41 tiles are exported to `results.csv`.

| Tile | Source | Variant | SR PSNR | Bicubic PSNR | ΔPSNR | SR SSIM | Bicubic SSIM | ΔSSIM |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| building_076747_mapbox | mapbox | clean | 40.84 | 39.30 | +1.54 | 0.9882 | 0.9886 | -0.0004 |
| building_076747_mapbox | mapbox | bicubic_ds | 41.21 | 39.16 | +2.05 | 0.9881 | 0.9877 | +0.0004 |
| building_076747_mapbox | mapbox | realistic | 36.09 | 35.18 | +0.92 | 0.9149 | 0.8957 | +0.0191 |
| building_1e44ba_google | google | clean | 28.48 | 27.67 | +0.82 | 0.9073 | 0.8924 | +0.0149 |
| building_1e44ba_google | google | bicubic_ds | 28.55 | 27.60 | +0.95 | 0.9143 | 0.8954 | +0.0188 |
| building_1e44ba_google | google | realistic | 26.44 | 25.98 | +0.45 | 0.7953 | 0.7773 | +0.0180 |
| building_20d961_google | google | clean | 26.50 | 25.69 | +0.82 | 0.5716 | 0.5756 | -0.0040 |
| building_20d961_google | google | bicubic_ds | 25.97 | 25.35 | +0.62 | 0.5557 | 0.5766 | -0.0209 |
| building_20d961_google | google | realistic | 25.90 | 25.15 | +0.75 | 0.5106 | 0.5022 | +0.0084 |
| building_22cade_mapbox | mapbox | clean | 41.22 | 40.22 | +1.00 | 0.9907 | 0.9934 | -0.0027 |
| building_22cade_mapbox | mapbox | bicubic_ds | 42.06 | 40.39 | +1.66 | 0.9910 | 0.9933 | -0.0023 |
| building_22cade_mapbox | mapbox | realistic | 36.00 | 35.39 | +0.61 | 0.9128 | 0.8939 | +0.0188 |
| building_3d9b33_google | google | clean | 26.88 | 25.65 | +1.23 | 0.8673 | 0.8411 | +0.0262 |
| building_3d9b33_google | google | bicubic_ds | 26.87 | 25.50 | +1.37 | 0.8757 | 0.8433 | +0.0324 |
| building_3d9b33_google | google | realistic | 25.48 | 24.67 | +0.81 | 0.7523 | 0.7238 | +0.0285 |
| building_418c44_mapbox | mapbox | clean | 42.54 | 41.50 | +1.04 | 0.9911 | 0.9933 | -0.0022 |
| building_418c44_mapbox | mapbox | bicubic_ds | 43.21 | 41.49 | +1.72 | 0.9913 | 0.9930 | -0.0018 |
| building_418c44_mapbox | mapbox | realistic | 36.67 | 36.02 | +0.66 | 0.9160 | 0.8983 | +0.0177 |
| building_43ebf7_mapbox | mapbox | clean | 36.21 | 33.97 | +2.24 | 0.9842 | 0.9803 | +0.0039 |
| building_43ebf7_mapbox | mapbox | bicubic_ds | 36.56 | 33.84 | +2.72 | 0.9832 | 0.9785 | +0.0047 |
| building_43ebf7_mapbox | mapbox | realistic | 33.11 | 31.85 | +1.26 | 0.8954 | 0.8734 | +0.0219 |
| building_4efaf9_mapbox | mapbox | clean | 40.47 | 39.26 | +1.21 | 0.9892 | 0.9901 | -0.0008 |
| building_4efaf9_mapbox | mapbox | bicubic_ds | 41.18 | 39.32 | +1.86 | 0.9896 | 0.9896 | -0.0000 |
| building_4efaf9_mapbox | mapbox | realistic | 35.23 | 34.83 | +0.40 | 0.9038 | 0.8855 | +0.0183 |
| building_513098_mapbox | mapbox | clean | 41.77 | 40.10 | +1.67 | 0.9907 | 0.9922 | -0.0014 |
| building_513098_mapbox | mapbox | bicubic_ds | 42.14 | 39.97 | +2.17 | 0.9909 | 0.9918 | -0.0009 |
| building_513098_mapbox | mapbox | realistic | 36.56 | 35.76 | +0.80 | 0.9123 | 0.8932 | +0.0191 |
| building_57c41d_mapbox | mapbox | clean | 43.26 | 40.73 | +2.53 | 0.9891 | 0.9873 | +0.0018 |
| building_57c41d_mapbox | mapbox | bicubic_ds | 43.49 | 40.80 | +2.68 | 0.9891 | 0.9873 | +0.0018 |
| building_57c41d_mapbox | mapbox | realistic | 36.87 | 35.78 | +1.08 | 0.9176 | 0.8973 | +0.0203 |
| building_57c7e8_mapbox | mapbox | clean | 41.84 | 41.05 | +0.80 | 0.9914 | 0.9930 | -0.0016 |
| building_57c7e8_mapbox | mapbox | bicubic_ds | 42.58 | 41.15 | +1.44 | 0.9917 | 0.9928 | -0.0011 |
| building_57c7e8_mapbox | mapbox | realistic | 36.11 | 35.55 | +0.56 | 0.9148 | 0.8972 | +0.0176 |
| building_5aca8b_google | google | clean | 28.98 | 28.59 | +0.39 | 0.8821 | 0.8797 | +0.0024 |
| building_5aca8b_google | google | bicubic_ds | 29.05 | 28.53 | +0.52 | 0.8910 | 0.8864 | +0.0046 |
| building_5aca8b_google | google | realistic | 27.48 | 27.25 | +0.24 | 0.7896 | 0.7745 | +0.0151 |
| building_666e65_google | google | clean | 28.30 | 27.61 | +0.69 | 0.9075 | 0.9020 | +0.0055 |
| building_666e65_google | google | bicubic_ds | 28.54 | 27.74 | +0.80 | 0.9151 | 0.9073 | +0.0078 |
| building_666e65_google | google | realistic | 26.73 | 26.35 | +0.38 | 0.7930 | 0.7766 | +0.0163 |
| building_686199_google | google | clean | 28.61 | 28.35 | +0.27 | 0.8324 | 0.8292 | +0.0031 |
| building_686199_google | google | bicubic_ds | 28.52 | 28.20 | +0.32 | 0.8415 | 0.8362 | +0.0054 |
| building_686199_google | google | realistic | 27.02 | 26.88 | +0.14 | 0.7268 | 0.7155 | +0.0113 |
| building_6c1ccc_mapbox | mapbox | clean | 41.57 | 39.57 | +2.01 | 0.9914 | 0.9916 | -0.0003 |
| building_6c1ccc_mapbox | mapbox | bicubic_ds | 41.97 | 39.44 | +2.52 | 0.9912 | 0.9909 | +0.0003 |
| building_6c1ccc_mapbox | mapbox | realistic | 36.71 | 35.55 | +1.16 | 0.9162 | 0.8961 | +0.0201 |
| building_6c8fd5_google | google | clean | 33.66 | 32.28 | +1.38 | 0.9588 | 0.9517 | +0.0070 |
| building_6c8fd5_google | google | bicubic_ds | 34.49 | 32.54 | +1.95 | 0.9647 | 0.9553 | +0.0094 |
| building_6c8fd5_google | google | realistic | 29.88 | 29.40 | +0.48 | 0.8467 | 0.8296 | +0.0171 |
| building_7345ba_google | google | clean | 25.97 | 25.73 | +0.24 | 0.8101 | 0.8086 | +0.0015 |
| building_7345ba_google | google | bicubic_ds | 26.04 | 25.81 | +0.23 | 0.8294 | 0.8293 | +0.0000 |
| building_7345ba_google | google | realistic | 24.97 | 24.81 | +0.16 | 0.7196 | 0.7074 | +0.0122 |
| building_761669_google | google | clean | 28.55 | 27.67 | +0.88 | 0.8689 | 0.8586 | +0.0103 |
| building_761669_google | google | bicubic_ds | 28.65 | 27.52 | +1.13 | 0.8751 | 0.8606 | +0.0145 |
| building_761669_google | google | realistic | 27.05 | 26.56 | +0.49 | 0.7588 | 0.7407 | +0.0181 |
| building_7d9f17_mapbox | mapbox | clean | 39.62 | 38.36 | +1.26 | 0.9892 | 0.9901 | -0.0009 |
| building_7d9f17_mapbox | mapbox | bicubic_ds | 40.39 | 38.44 | +1.95 | 0.9896 | 0.9897 | -0.0001 |
| building_7d9f17_mapbox | mapbox | realistic | 35.21 | 34.63 | +0.58 | 0.9097 | 0.8911 | +0.0186 |
| building_7ecfdc_google | google | clean | 26.39 | 25.64 | +0.75 | 0.8741 | 0.8657 | +0.0083 |
| building_7ecfdc_google | google | bicubic_ds | 26.40 | 25.76 | +0.64 | 0.8830 | 0.8762 | +0.0068 |
| building_7ecfdc_google | google | realistic | 24.69 | 24.20 | +0.49 | 0.7618 | 0.7521 | +0.0097 |
| building_814230_mapbox | mapbox | clean | 38.92 | 37.36 | +1.57 | 0.9855 | 0.9856 | -0.0001 |
| building_814230_mapbox | mapbox | bicubic_ds | 39.51 | 37.28 | +2.23 | 0.9864 | 0.9851 | +0.0013 |
| building_814230_mapbox | mapbox | realistic | 34.59 | 33.93 | +0.66 | 0.9074 | 0.8913 | +0.0161 |
| building_8899da_google | google | clean | 31.13 | 28.55 | +2.58 | 0.9540 | 0.9342 | +0.0198 |
| building_8899da_google | google | bicubic_ds | 31.56 | 28.60 | +2.96 | 0.9570 | 0.9342 | +0.0228 |
| building_8899da_google | google | realistic | 28.02 | 26.69 | +1.33 | 0.8249 | 0.8001 | +0.0248 |
| building_8aeff0_mapbox | mapbox | clean | 43.98 | 42.47 | +1.52 | 0.9936 | 0.9961 | -0.0025 |
| building_8aeff0_mapbox | mapbox | bicubic_ds | 44.51 | 42.47 | +2.03 | 0.9935 | 0.9958 | -0.0023 |
| building_8aeff0_mapbox | mapbox | realistic | 37.92 | 36.99 | +0.93 | 0.9188 | 0.8988 | +0.0200 |
| building_8d05f2_google | google | clean | 34.61 | 33.28 | +1.33 | 0.9694 | 0.9660 | +0.0033 |
| building_8d05f2_google | google | bicubic_ds | 34.74 | 33.30 | +1.44 | 0.9704 | 0.9658 | +0.0046 |
| building_8d05f2_google | google | realistic | 32.03 | 31.14 | +0.88 | 0.8842 | 0.8625 | +0.0216 |
| building_90435e_google | google | clean | 23.19 | 22.63 | +0.56 | 0.7433 | 0.7219 | +0.0214 |
| building_90435e_google | google | bicubic_ds | 23.32 | 22.55 | +0.77 | 0.7694 | 0.7403 | +0.0291 |
| building_90435e_google | google | realistic | 22.29 | 21.96 | +0.33 | 0.6469 | 0.6260 | +0.0209 |
| building_98fa77_google | google | clean | 28.03 | 27.33 | +0.70 | 0.9364 | 0.9304 | +0.0059 |
| building_98fa77_google | google | bicubic_ds | 28.74 | 27.90 | +0.84 | 0.9471 | 0.9399 | +0.0072 |
| building_98fa77_google | google | realistic | 26.31 | 25.94 | +0.38 | 0.8222 | 0.8061 | +0.0161 |
| building_a3a517_google | google | clean | 27.48 | 26.94 | +0.54 | 0.8295 | 0.8268 | +0.0028 |
| building_a3a517_google | google | bicubic_ds | 27.57 | 26.92 | +0.66 | 0.8382 | 0.8333 | +0.0049 |
| building_a3a517_google | google | realistic | 26.42 | 26.06 | +0.36 | 0.7458 | 0.7316 | +0.0142 |
| building_a90908_google | google | clean | 26.00 | 25.79 | +0.21 | 0.8102 | 0.8082 | +0.0020 |
| building_a90908_google | google | bicubic_ds | 26.05 | 25.86 | +0.19 | 0.8295 | 0.8288 | +0.0006 |
| building_a90908_google | google | realistic | 25.00 | 24.86 | +0.14 | 0.7237 | 0.7117 | +0.0120 |
| building_aba1eb_google | google | clean | 29.72 | 28.18 | +1.54 | 0.9215 | 0.8975 | +0.0241 |
| building_aba1eb_google | google | bicubic_ds | 30.10 | 28.19 | +1.91 | 0.9297 | 0.9020 | +0.0277 |
| building_aba1eb_google | google | realistic | 26.97 | 26.20 | +0.77 | 0.7827 | 0.7601 | +0.0226 |
| building_b31756_google | google | clean | 29.65 | 28.49 | +1.16 | 0.9074 | 0.8908 | +0.0166 |
| building_b31756_google | google | bicubic_ds | 29.78 | 28.40 | +1.38 | 0.9140 | 0.8940 | +0.0200 |
| building_b31756_google | google | realistic | 27.33 | 26.71 | +0.62 | 0.7810 | 0.7599 | +0.0212 |
| building_b31756_mapbox | mapbox | clean | 39.79 | 37.64 | +2.15 | 0.9863 | 0.9858 | +0.0005 |
| building_b31756_mapbox | mapbox | bicubic_ds | 40.10 | 37.59 | +2.51 | 0.9859 | 0.9848 | +0.0011 |
| building_b31756_mapbox | mapbox | realistic | 35.27 | 34.11 | +1.16 | 0.9081 | 0.8894 | +0.0187 |
| building_c85563_google | google | clean | 31.20 | 29.31 | +1.89 | 0.9633 | 0.9504 | +0.0129 |
| building_c85563_google | google | bicubic_ds | 31.66 | 29.50 | +2.16 | 0.9668 | 0.9517 | +0.0151 |
| building_c85563_google | google | realistic | 28.36 | 27.41 | +0.96 | 0.8552 | 0.8349 | +0.0203 |
| building_c85563_mapbox | mapbox | clean | 39.64 | 37.56 | +2.08 | 0.9907 | 0.9897 | +0.0010 |
| building_c85563_mapbox | mapbox | bicubic_ds | 40.00 | 37.39 | +2.61 | 0.9902 | 0.9886 | +0.0017 |
| building_c85563_mapbox | mapbox | realistic | 35.57 | 34.43 | +1.14 | 0.9114 | 0.8920 | +0.0195 |
| building_cac04a_google | google | clean | 27.68 | 25.72 | +1.96 | 0.9114 | 0.8821 | +0.0293 |
| building_cac04a_google | google | bicubic_ds | 27.99 | 25.67 | +2.31 | 0.9151 | 0.8842 | +0.0309 |
| building_cac04a_google | google | realistic | 26.25 | 24.73 | +1.53 | 0.8134 | 0.7774 | +0.0360 |
| building_d87dbc_google | google | clean | 26.69 | 25.91 | +0.78 | 0.8830 | 0.8685 | +0.0144 |
| building_d87dbc_google | google | bicubic_ds | 26.91 | 25.94 | +0.97 | 0.8914 | 0.8729 | +0.0185 |
| building_d87dbc_google | google | realistic | 25.04 | 24.65 | +0.39 | 0.7704 | 0.7506 | +0.0198 |
| building_dc7b9e_mapbox | mapbox | clean | 40.22 | 39.10 | +1.12 | 0.9880 | 0.9907 | -0.0027 |
| building_dc7b9e_mapbox | mapbox | bicubic_ds | 41.16 | 39.41 | +1.75 | 0.9886 | 0.9909 | -0.0022 |
| building_dc7b9e_mapbox | mapbox | realistic | 35.44 | 34.83 | +0.61 | 0.9097 | 0.8920 | +0.0177 |
| building_e42da1_google | google | clean | 24.85 | 24.45 | +0.39 | 0.7553 | 0.7422 | +0.0131 |
| building_e42da1_google | google | bicubic_ds | 24.66 | 24.25 | +0.41 | 0.7661 | 0.7501 | +0.0159 |
| building_e42da1_google | google | realistic | 23.78 | 23.49 | +0.29 | 0.6520 | 0.6381 | +0.0138 |
| building_f28a5f_google | google | clean | 29.09 | 27.67 | +1.42 | 0.8666 | 0.8544 | +0.0122 |
| building_f28a5f_google | google | bicubic_ds | 29.59 | 27.67 | +1.92 | 0.8871 | 0.8674 | +0.0197 |
| building_f28a5f_google | google | realistic | 27.25 | 26.43 | +0.82 | 0.7357 | 0.7171 | +0.0187 |
| building_f73de6_google | google | clean | 32.64 | 31.82 | +0.82 | 0.9071 | 0.9061 | +0.0010 |
| building_f73de6_google | google | bicubic_ds | 32.72 | 31.94 | +0.78 | 0.9129 | 0.9126 | +0.0003 |
| building_f73de6_google | google | realistic | 29.80 | 29.28 | +0.51 | 0.7930 | 0.7828 | +0.0102 |
| building_f73de6_mapbox | mapbox | clean | 43.34 | 41.80 | +1.53 | 0.9923 | 0.9926 | -0.0003 |
| building_f73de6_mapbox | mapbox | bicubic_ds | 43.43 | 41.63 | +1.79 | 0.9919 | 0.9921 | -0.0002 |
| building_f73de6_mapbox | mapbox | realistic | 36.99 | 36.13 | +0.86 | 0.9213 | 0.9052 | +0.0161 |
| building_fc48ee_mapbox | mapbox | clean | 43.98 | 42.81 | +1.16 | 0.9926 | 0.9941 | -0.0016 |
| building_fc48ee_mapbox | mapbox | bicubic_ds | 44.31 | 42.66 | +1.65 | 0.9924 | 0.9937 | -0.0013 |
| building_fc48ee_mapbox | mapbox | realistic | 37.19 | 36.49 | +0.70 | 0.9176 | 0.8999 | +0.0177 |
