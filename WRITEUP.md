# Satellite Tile Enhancement: Technical Report

## 1. Headline Answers and Pipeline Rationale
* **SR vs Bicubic:** FSRCNN 2x beats bicubic: **+0.68 dB PSNR** (95% CI: [+0.58, +0.78], SSIM +0.018) on blur/noise/JPEG (100% win rate; Google-only: **+0.57 dB**). The bicubic_ds gain (+1.53 dB) reflects ~0.32 dB training-kernel bias.
* **Classical Pipeline:** Globally degrades real tiles. On blur/noise, PSNR collapses **-11.27 dB** (SSIM -0.17), aiding only haze (+3.42 dB). On real tiles, it amplifies noise 2.8x (0.60 to 1.70) and shifts Delta E to 27.9. Contrast causes this (contrast_only: -10.26 dB); denoise_only is the sole helpful stage (+0.88 dB).
* **Ordering:** NLM denoising (h=3), CIELAB stretch + CLAHE (clipLimit=2.0, 8x8 grid), then unsharp masking (sigma=1.0, threshold 3.0). White balance is omitted. Flaw: 8x8 CLAHE on tiny tiles (75x103) yields ~10px cells that over-amplify contrast.

## 2. Evaluation Design, Biases, and Reference Strategy
Native tiles (N=41) were degraded with blur, noise (sigma=3), JPEG (Q=60-70), and Koschmieder haze (t=0.7, A=220). Google scale=2 optical pairs were considered but omitted to avoid API keys and rate limits; moreover, scale=2 may return interpolated pixels rather than genuine sensor MTF.
* **Biases & Disclosures:** Mapbox tiles inflate metrics (clean PSNR ~40 dB vs Google baseline). Full-reference metrics penalize valid tone stretching. Sobel correlation is vacuous for affine haze (0.9996). Clean tiles yield 17.9 dB enhanced PSNR. Laplacian variance falsely rewards noise over sharpness.

## 3. Three Failure Cases on Google Tiles
Three failure modes on Google aerial tiles (Case 1 isolated by ablation; [crops in failures/](failures/)):
1. **Luminance Blowout** ([case 1](failures/case1_contrast_aba1eb.png)): On building_aba1eb_google, Delta E reached 55.82. Contrast_only dropped PSNR by **-14.54 dB** (25.77 to 11.23 dB; denoise_only preserved ~26 dB). Stretch and CLAHE stacked two contrast boosts, clipping roof highlights.
2. **Noise Amplification** ([case 2](failures/case2_noise_e42da1.png)): On building_e42da1_google, noise sigma surged from 4.79 to 14.19; Laplacian variance exploded 7.6x (3,822 to 29,121). Lifting shadows likely exposed sensor grain that sharpening amplified into visible noise.
3. **Minor SSIM Regression** ([case 3](failures/case3_sr_ssim_20d961.png)): On building_20d961_google, FSRCNN beat bicubic on PSNR (+0.62 to +0.82 dB) but had minor SSIM regression (-0.021 on bicubic_ds, -0.004 on clean) on periodic rooflines.

## 4. 50,000 Tiles/Day Throughput & Real Bottlenecks
At 50,000 tiles/day (~12.2 core-hours), CPU compute is not the primary bottleneck:
1. **API Quota & Cost:** Ingesting 50k tiles/day (1.5M/mo) may trigger severe rate limits and query fees.
2. **Storage Compliance:** Persisting tiles in a database conflicts with provider caching terms.
3. **I/O & Concurrency:** Network fetches and uncompressed buffers saturate bandwidth and RAM.
4. **Algorithmic Scaling:** NLM takes >95% of classical CPU time; on large tiles (>1000x1000), latency scales linearly (~2s). Replace NLM with guided filtering (<15ms) and export FSRCNN to ONNX Runtime (<65ms).

## 5. Google Maps Platform Terms of Service
Storing enhanced satellite tiles in a database to serve a product appears to conflict with Google Maps Platform Terms of Service. Under headings **"No Scraping"** and **"No Creating Content From Google Maps Content"**, Google prohibits pre-fetching, bulk downloading, and creating derivative works. Caching is strictly limited under **Service Specific Terms** to temporary performance caching, not persistent storage. Imagery must appear exclusively within authorized Google Maps Core Services, preserving all logos and copyright notices.
**Do Not:** Store tiles in a database, strip watermarks, or serve tiles outside Google Maps SDKs without written agreement. Obtain legal review before deployment.
