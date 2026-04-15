# embed2heights Label Band Analysis Report

## 1. Dataset Overview

- **Source**: French IGN airborne LiDAR products, native 1 m resolution aggregated to 10 m pixels
- **Scale**: 2,024 patches, each a 4-band GeoTIFF
- **Spatial size**: Nominal 256x256, but **258 samples (12.7%) are 255x256 or 256x255** — training pipeline must pad or resize
- **Band definitions**:

| Band | Description | Range | Unit |
|------|-------------|-------|------|
| 0 | Building fraction | [0, 1] | area fraction |
| 1 | Vegetation fraction | [0, 1] | area fraction |
| 2 | Water fraction | [0, 1] | area fraction |
| 3 | nDSM height | [0, 209.9] | meters |

## 2. Data Quality

### 2.1 Numerical Cleanliness: PASS

- All 4 bands are free of NaN, Inf, and negative values
- Percentage bands (0–2) are strictly within [0, 1] with zero out-of-range pixels
- Sum of percentage bands has **max = 1.0** — no pixel exceeds 1, mutual exclusivity is perfect

### 2.2 Nodata: Severe and Pervasive

Two distinct types of zero-valued regions exist in the data:

| Type | Description | Affected Samples | Mean Fraction |
|------|-------------|------------------|---------------|
| **True Nodata** | All 4 bands = 0; patch extends beyond imagery boundary | 82.7% (1,673 / 2,024) | 39% of all pixels |
| **nDSM-only Missing** | nDSM = 0 but land cover bands have valid values; nDSM source lacks coverage | 22.6% (458 / 2,024) | 1.8% of all pixels |

- 32.7% of samples have more than half their area as nodata
- Rectangular nDSM holes (contiguous row/column run >= 16) appear in **15.7% (317 / 2,024)** of samples
- Of these, 135 samples have run >= 64 and 48 samples have run >= 128

## 3. Class Distribution and Imbalance

### 3.1 Pixel-level Distribution

| Band | Non-zero Pixels | Mean | Std | Median | P99 |
|------|----------------|------|-----|--------|-----|
| Building | **3.3%** | 0.012 | 0.087 | 0.000 | 0.51 |
| Vegetation | 40.7% | 0.363 | 0.469 | 0.000 | 1.00 |
| Water | **1.7%** | 0.014 | 0.116 | 0.000 | 1.00 |
| nDSM | 59.2% | 3.87 m | 6.35 m | 0.095 m | 24.21 m |

### 3.2 Sample-level Distribution

| Band | Samples >95% Zero | Imbalance Severity |
|------|-------------------|--------------------|
| Building | 82.6% (1,671 / 2,024) | **Extreme** |
| Vegetation | 15.2% (307 / 2,024) | Low |
| Water | 93.3% (1,888 / 2,024) | **Extreme** |

- Building and Water are extremely sparse — virtually absent in most patches
- 61.3% of pixels are "Bare/Other" (all three classes = 0), corresponding to the LiDAR Background class which is not labeled separately

## 4. nDSM (Band 3) Verification

### 4.1 Physical Plausibility: PASS

- Range [0, 209.9 m], no negative values
- 99.9% of values < 30 m, consistent with expected building and tree heights
- Outliers: 469 pixels > 100 m (0.0004%), likely LiDAR noise or tall structures (towers, chimneys)

### 4.2 Correlation with Land Cover: Good

nDSM shows clear separability across land cover classes:

| Land Cover | nDSM Mean | nDSM Median | [P5, P95] | Physical Interpretation |
|------------|-----------|-------------|-----------|------------------------|
| **Vegetation** | 9.54 m | 8.71 m | [0.02, 22.4] m | Canopy height |
| **Building** | 5.82 m | 4.56 m | [1.8, 14.0] m | Building story height |
| **Water** | 1.77 m | 0.01 m | [0.0, 9.2] m | Near ground level; some bridges/levees |
| **Bare/Other** | 0.54 m | 0.00 m | [0.0, 3.3] m | Ground surface |

- Vegetation and Building height distributions overlap but remain distinguishable (median gap ~4 m)
- Water and Bare are close to ground level with limited height information
- **Conclusion: nDSM provides meaningful physical signal and can be used directly for model training**

### 4.3 Signal Quality

- 13.5% of samples have nDSM std < 0.5 m (nearly flat, no height variation)
- 17.1% of samples have nDSM mean < 0.5 m (large flat areas or nodata-dominated)
- 48.2% of samples have nDSM max > 30 m

## 5. Recommendations for Model Design

### 5.1 Critical: Nodata Masking

This is the highest-priority fix:

1. **Global validity mask**: `valid = NOT (band0==0 AND band1==0 AND band2==0 AND band3==0)` — all loss terms must exclude these pixels
2. **nDSM-specific mask**: `height_valid = valid AND NOT (ndsm==0 AND (band0>0 OR band1>0 OR band2>0))` — height loss must additionally exclude nDSM-missing regions
3. **258 non-standard-shape samples**: Zero-pad to 256x256 and include padded regions in the nodata mask

> If the baseline does not implement these masks, ~39% of pixels contribute incorrect gradients (training on nodata=0 as ground truth). Fixing this alone is expected to directly improve RMSE.

### 5.2 Class Imbalance Mitigation

The extreme sparsity of Building (3.3%) and Water (1.7%) requires aggressive strategies:

- **Loss reweighting**: Increase loss weight for Building and Water channels (current baseline Tversky alpha=0.3, beta=0.7 may be insufficient)
- **Sample oversampling**: Oversample patches containing building/water pixels
- **Focal strategy**: Use focal loss variants for sparse classes to down-weight easy negatives

### 5.3 nDSM Normalization

- Baseline uses `/30.0` normalization. Given P99 = 24.2 m, **30 m is a reasonable normalization constant**
- However, max = 209.9 m suggests **clipping to [0, 45 m]** (~P99.95) to prevent extreme values from distorting training
- nDSM distribution is heavily right-skewed (median = 0.095 m vs. mean = 3.87 m) — consider **log(1+x) transform** for more uniform distribution
- No negative values exist; no additional handling needed

### 5.4 Percentage Band Characteristics

- The three percentage bands are mutually exclusive with sum <= 1, but **do not apply softmax** — many pixels have all three classes at 0 (bare ground); this is not a 3-class classification problem
- Vegetation is near-binary (concentrated at 0 and 1); Building and Water are more continuous but extremely sparse
- No additional normalization needed; values are already in [0, 1]

### 5.5 Evaluation Metric Alignment

| Metric | Weight | Depends On | Key Challenge |
|--------|--------|------------|---------------|
| mIoU_buildings | 25% | Band 0 | Extreme sparsity (3.3%) |
| RMSE_building_height | 25% | Band 0 + Band 3 | Sparsity + nodata |
| RMSE_vegetation_height | 20% | Band 1 + Band 3 | nDSM missing regions |
| mIoU_trees | 15% | Band 1 | Moderate distribution |
| mIoU_water | 15% | Band 2 | Extreme sparsity (1.7%) |

- **50% of the score depends on Building-related metrics**, yet Building occupies only 3.3% of pixels — this is where effort should be concentrated
- **45% of the score depends on height RMSE** — correct nodata masking directly impacts nearly half the total score
