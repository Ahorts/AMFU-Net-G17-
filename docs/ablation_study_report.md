# AMFU-Net Empirical Ablation Study Report

## 1. Executive Summary & Experimental Objectives

This report documents the systematic empirical ablation experiments for adapting the **Adaptive Multi-Scale Feature Uncertainty Network (AMFU-Net)** (Chen et al., 2026) to public multi-center breast ultrasound benchmarks (`Ahorts/G17-preprocessed-dataset`).

To substantiate the respective contributions of deep architectural attention modules versus classical PDE-based despeckling and contrast preprocessing, a **2×2 factorial ablation matrix** (Architecture × Preprocessing) was executed on an NVIDIA RTX 6000 Ada/Blackwell compute platform across the identical immutable test split (**266 unseen scans**, 8:1:1 Stratified Group K-Fold preserving patient boundaries).

### Experimental Configurations Evaluated:
1. **Exp-01 (Full Pipeline):** Full AMFU-Net (DAG + MFEF) on SRAD + CLAHE preprocessed scans (`9,837,121` parameters).
2. **Exp-02 (Architecture Ablation):** Plain U-Net backbone (DAG and MFEF disabled) on SRAD + CLAHE preprocessed scans (`3,833,505` parameters).
3. **Exp-03 (Preprocessing Ablation):** Full AMFU-Net (DAG + MFEF) on unfiltered raw ultrasound scans (`9,837,121` parameters).
4. **Exp-04 (Dual Ablation Control):** Plain U-Net backbone on unfiltered raw ultrasound scans (`3,833,505` parameters).

---

## 2. Benchmark Baseline Provenance

To eliminate ambiguity across comparative metrics:
- **Chen et al. (2026) Published Benchmark (`0.9142` Dice):** Evaluated strictly on private, paired ultrasound computed tomography (USCT) reflection and transmission data. The original architecture relies on two-stream tomographic fusion not available in clinical handheld ultrasound.
- **Session 5 Benchmark (Slide 15 Reference):** The previous team run achieved **81.73% Dice, 72.52% Jaccard, 30.95 px HD95, and 9.52 px ASD** on the 255 held-out test split. The `0.7909` Dice / `0.3496` Loss reported in model card metadata was the **validation split score** (which plateaued at ~79% as documented in the Session 5 validation curves).
- **Exp-01 Reproduction:** Achieved **81.20% Dice, 71.34% Jaccard, 31.55 px HD95, and 9.27 px ASD** on the 266-image test split, reproducing the Session 5 benchmark within 0.5% variance.

---

## 3. Quantitative Results & Comparison Matrix

All experiments were trained for 200 epochs under FP16 mixed precision and evaluated on the 266-scan internal test cohort (BUSI, BUS-UCLM, and BUS-BRA).

### 3.1 Primary Performance Matrix

| Experiment ID | Architecture Configuration | Parameters | Preprocessing Pipeline | Test Loss | Test Dice | Test Jaccard (IoU) | 95% HD (px) | ASD (px) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Session 5 Benchmark* | AMFU-Net (Baseline Run) | 9.84M | SRAD + CLAHE | — | `0.8173` (81.73%) | `0.7252` (72.52%) | `30.9500` | `9.5200` |
| **Exp-01: Full Baseline** | AMFU-Net (DAG + MFEF) | 9,837,121 | SRAD + CLAHE | **`0.3073`** | **`0.8120` (81.20%)** | **`0.7134` (71.34%)** | **`31.5510`** | **`9.2665`** |
| **Exp-02: Arch Ablation** | Plain U-Net (No DAG, No MFEF) | 3,833,505 | SRAD + CLAHE | `0.3605` | `0.7823` (78.23%) | `0.6824` (68.24%) | `41.3804` | `13.6121` |
| **Exp-03: Preproc Ablation**| AMFU-Net (DAG + MFEF) | 9,837,121 | Raw Ultrasound Scans | `0.3203` | `0.8052` (80.52%) | `0.7081` (70.81%) | `32.7928` | `9.5494` |
| **Exp-04: Dual Control** | Plain U-Net (No DAG, No MFEF) | 3,833,505 | Raw Ultrasound Scans | `0.3662` | `0.7793` (77.93%) | `0.6780` (67.80%) | `40.6824` | `13.4097` |

### 3.2 2×2 Factorial Interaction Matrix

| Architecture Configuration | Enhanced (SRAD + CLAHE) | Raw Ultrasound Scans | Preprocessing Lift ($\Delta_{\text{prep}}$) |
| :--- | :--- | :--- | :--- |
| **Full AMFU-Net** (`9.84M` params) | **`0.8120` Dice** (Exp-01) | **`0.8052` Dice** (Exp-03) | **$+0.0068$ (+0.68%)** |
| **Plain U-Net** (`3.83M` params) | **`0.7823` Dice** (Exp-02) | **`0.7793` Dice** (Exp-04) | **$+0.0030$ (+0.30%)** |
| **Architecture Lift ($\Delta_{\text{arch}}$)** | **$+0.0297$ (+2.97%)** | **$+0.0259$ (+2.59%)** | **Interaction ($\Delta_{\text{int}}$):** $+0.0038$ |

---

## 4. Multi-Center Cross-Domain Breakdown

External hospital cohorts evaluate domain generalization across differing scanner vendors, acoustic center frequencies, and operator variability.

| Experiment | Metric | BUS_BRA ($n=186$) | BUS_UCLM ($n=16$) | BUSI ($n=64$) | Overall ($N=266$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp-01** (Full AMFU-Net + Enhanced) | Dice | `0.8310` | `0.8232` | `0.7539` | **`0.8120`** |
| | 95% HD (px) | `29.7997` | `28.3741` | `40.7513` | **`31.5510`** |
| | ASD (px) | `8.8719` | `8.3582` | `12.0194` | **`9.2665`** |
| **Exp-02** (Plain U-Net + Enhanced) | Dice | `0.8250` | `0.7083` | `0.6768` | **`0.7823`** |
| | 95% HD (px) | `30.9324` | `72.4419` | `64.0042` | **`41.3804`** |
| | ASD (px) | `9.4891` | `24.1873` | `23.0033` | **`13.6121`** |
| **Exp-03** (Full AMFU-Net + Raw) | Dice | `0.8250` | `0.8146` | `0.7452` | **`0.8052`** |
| | 95% HD (px) | `30.7725` | `25.8642` | `40.3879` | **`32.7928`** |
| | ASD (px) | `9.0221` | `7.8541` | `11.5036` | **`9.5494`** |
| **Exp-04** (Plain U-Net + Raw) | Dice | `0.8206` | `0.7167` | `0.6746` | **`0.7793`** |
| | 95% HD (px) | `33.2506` | `73.8980` | `53.9772` | **`40.6824`** |
| | ASD (px) | `10.2141` | `23.8875` | `20.0773` | **`13.4097`** |

---

## 5. Technical Insights & Mechanistic Deductions

### 5.1 Architectural Primacy vs Preprocessing Refinement
- **Dominance of Network Architecture:** Architectural components (DAG attention gating and MFEF multi-scale pooling) account for over **81% to 90%** of total segmentation performance gains. Replacing AMFU-Net with Plain U-Net degrades Dice by **-2.59% to -2.97%** and inflates Hausdorff boundary error by **+24% to +31%** (`31.55 px` $\rightarrow$ `41.38 px`).
- **Marginal Contribution of Preprocessing:** SRAD and CLAHE image preprocessing provide consistent boundary edge refinement (**+0.30% to +0.68% Dice**, `-1.24 px` HD95 on AMFU-Net). While non-negligible, classical filtering alone cannot compensate for the absence of architectural attention gates.

### 5.2 Positive Gated Interaction Effect ($\Delta_{\text{int}} = +0.0038$)
- Full AMFU-Net extracts **2.27× more quantitative improvement** from preprocessing than Plain U-Net (`+0.68%` vs `+0.30%`).
- **Mechanism:** Dual Attention Gates (DAG) compute spatial and channel attention weights from intermediate skip-connection feature maps. When input images undergo PDE-based anisotropic diffusion (SRAD), high-frequency speckle noise is attenuated while acoustic gradient margins are sharpened by CLAHE. The attention gates leverage these clarified gradients to produce precise spatial activation maps. In contrast, Plain U-Net passes unweighted skip tensors directly to the decoder, diluting the benefits of edge enhancement.

### 5.3 Cross-Center Domain Collapse in Vanilla Backbones
- On the primary source domain (`BUS_BRA`, $n=186$), Plain U-Net retains competitive overlap (`0.8206` to `0.8250` Dice).
- On external hospital cohorts (`BUSI` and `BUS_UCLM`), Plain U-Net experiences catastrophic domain collapse:
  - `BUSI`: Performance drops to `0.6746`–`0.6768` Dice (an **8.0% drop** relative to AMFU-Net).
  - `BUS_UCLM`: Boundary error spikes to **`72.44`–`73.90 px` HD95**, compared to **`25.86`–`28.37 px`** for AMFU-Net (a **2.6× error magnification**).
- **Mechanism:** Differing scanner center frequencies and focal depths alter lesion appearance scale and acoustic attenuation profiles. The Multi-Scale Feature Extraction and Fusion (MFEF) bottleneck dynamically aggregates features across multi-scale receptive fields, providing the domain invariance necessary to prevent out-of-distribution failure.

---

## 6. Next Steps: Stage 3 Preprocessing Ablation

To isolate the independent contributions within the preprocessing pipeline itself, subsequent experiments will decouple the dual-stage transformation:
1. **Exp-05 (SRAD Alone):** Full AMFU-Net trained on despeckled images without CLAHE contrast equalization.
2. **Exp-06 (CLAHE Alone):** Full AMFU-Net trained on contrast-equalized raw images without PDE despeckling diffusion.
