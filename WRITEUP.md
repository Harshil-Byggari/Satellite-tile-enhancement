# Satellite Tile Enhancement: Technical Report

## 1. Headline Answers and Pipeline Rationale
* **SR vs Bicubic:** FSRCNN 2x beats bicubic: **+0.68 dB PSNR** (95% CI: [+0.58, +0.78], SSIM +0.018) under realistic blur/noise/JPEG (100% win rate); Google-only gain is **+0.57 dB**. The bicubic_ds gain (+1.53 dB) reflects ~0.32 dB training-kernel bias over clean.
* **Classical Pipeline:** Globally hurts real tiles. On blur/noise, PSNR collapses by **-11.27 dB** (SSIM -0.17); it only improves under haze (+3.42 dB). On real tiles, it raises noise 2.8x (0.60 to 1.70), sharpness 6.2x, and mean Delta E to 27.9 ('radioactive' saturation). Contrast expansion is responsible (contrast_only costs -10.26 dB); denoise_only is the sole stage that helps (+0.88 dB).
* **Ordering:** Low-to-high frequency: NLM denoising (h=3), CIELAB stretch + CLAHE (clipLimit=2.0, 8x8 grid), and unsharp masking (sigma=1.0, threshold 3.0). White balance is omitted. Flaw: an 8x8 CLAHE grid on small tiles (75x103) yields tiny ~10px cells that over-amplify contrast.

## 2. Evaluation Design, Biases, and Reference Strategy
Ground truth was synthesized via downsampling, blur, noise (sigma=3.0), JPEG (Q=60/65/70), and Koschmieder optical haze (t=0.7, A=220) on native tiles (N=41). We considered querying Google Static Maps scale=2 for optical pairs, but bypassed this to avoid live API billing, quotas, and key dependencies that break offline reproducibility; furthermore, scale=2 often returns interpolated pixels rather than genuine sensor MTF.
* **Biases & Disclosures:** Mapbox tiles inflate aggregates (clean PSNR ~40 dB, Laplacian ~168, pre-smoothed data); Google-only gives the true baseline. Three tile pairs share identical coordinates. Full-reference metrics penalize contrast stretching. Sobel correlation is vacuous for haze (0.9996) because haze is an affine transform. On clean tiles, absolute enhanced PSNR is 17.9 dB (capping delta artifact -82 dB). Laplacian variance is biased: it rewards noise amplification over sharpness.

## 3. Three Failure Cases on Google Tiles
Stage ablations isolate three failure modes on Google aerial tiles ([crops in failures/](failures/)):
1. **Luminance Blowout** ([case 1](failures/case1_contrast_aba1eb.png)): On building_aba1eb_google, Delta E reached 55.82. Stage ablation confirms contrast_only collapsed PSNR by **-14.54 dB** (25.77 to 11.23 dB; denoise_only preserved ~26 dB). Percentile stretch and CLAHE stacked two contrast boosts, blowing out roof highlights.
2. **Noise Amplification** ([case 2](failures/case2_noise_e42da1.png)): On building_e42da1_google, wavelet noise sigma surged from 4.79 to 14.19 (+9.40); Laplacian variance exploded 7.6x (3,822 to 29,121). Lifting shadow floors likely exposed sensor grain that unsharp masking amplified into visible noise, which Laplacian variance falsely rewarded as sharpness.
3. **Minor SSIM Regression** ([case 3](failures/case3_sr_ssim_20d961.png)): On building_20d961_google, FSRCNN 2x improved PSNR (+0.62 to +0.82 dB) but suffered a minor SSIM regression against bicubic (-0.021 on bicubic_ds, -0.004 on clean) across dense repeating roof patterns.

## 4. 50,000 Tiles/Day Throughput & Real Bottlenecks
Single-core CPU benchmarks on 640x640 tiles: Classical is 0.88s/tile (NLM consumes 0.85s or 95.5%); FSRCNN is 0.60s/tile; Bicubic is 0.004s/tile.
At 50k tiles/day (~12.2 core-hours), CPU compute is not what breaks first. Primary bottlenecks:
1. **API Quota & Cost:** Ingesting 50k tiles/day (1.5M/mo) triggers severe rate limits and query fees.
2. **Storage Compliance:** Persisting tiles in a database conflicts with provider caching terms.
3. **I/O & Memory:** Network fetches and uncompressed buffers saturate RAM under concurrent workers.
4. **Algorithmic Scaling:** NLM consumes >95% of classical CPU time; on high-resolution tiles (>1000x1000), latency scales linearly with pixel count to ~2s. Replace NLM with fast guided filtering (<15ms est.) and export FSRCNN to ONNX Runtime (<65ms est.).

## 5. Google Maps Platform Terms of Service
Storing enhanced satellite tiles in a database to serve a product appears to conflict with the Google Maps Platform Terms of Service. Under clause headings **"No Scraping"** and **"No Creating Content From Google Maps Content"**, Google prohibits pre-fetching, bulk downloading, and creating derivative works or modified imagery. Caching is strictly limited under the **Service Specific Terms** to temporary, ephemeral client-side caching to improve performance, rather than creating an independent database of tiles. Furthermore, imagery must be displayed exclusively within authorized Google Maps Core Services (Maps SDK / JS API), with all Google logos and copyright notices displayed without modification.
**Do Not:** (1) Permanently store tiles in a database; (2) strip Google watermarks or copyright notices; (3) display tiles outside authorized Google Maps SDKs; (4) feed imagery into machine learning pipelines without Google's written agreement. Always obtain formal legal review before deployment.
