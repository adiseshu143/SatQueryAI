# SatQuery AI — Remote Sensing Dataset & Benchmark Integration

## Overview

SatQuery AI natively integrates state-of-the-art open-source remote-sensing vision-language benchmark datasets for model training, evaluation, and zero-shot VQA verification:

1. **BigEarthNet-MM (`https://txt.bigearth.net/`)**
   - **Repository / Website:** [BigEarthNet Official Site](https://txt.bigearth.net/)
   - **Capabilities:** Large-scale multimodal benchmark featuring Sentinel-1 SAR (Radar Backscatter VV/VH) and Sentinel-2 Optical (Multispectral 12-band) satellite image patches with 19-class & 43-class CORINE Land Cover (CLC) multi-label annotations.
   - **Integration:** Joint SAR-optical radar penetration analysis, NDVI/NDWI spectral indices, and CORINE multi-label land cover tagging.
   - **Evaluation Script:** `python -m eval.evaluate_bigearthnet 5`

2. **CDVQA (`https://github.com/YZHJessica/CDVQA`)**
   - **Repository:** [CDVQA GitHub](https://github.com/YZHJessica/CDVQA)
   - **Capabilities:** Bi-temporal change detection visual question answering engine for comparative satellite analysis, SSIM similarity, and change magnitude categorization.
   - **Evaluation Script:** `python -m eval.evaluate_cdvqa 5`

3. **Zenodo RS VQA (`https://doi.org/10.5281/zenodo.6344366`)**
   - **DOI / Repository:** [Zenodo RS VQA Project](https://doi.org/10.5281/zenodo.6344366)
   - **Capabilities:** Remote-Sensing Vision-Language Model question-answering benchmark supporting Count queries, Presence queries, Area queries, and Comparison queries.
   - **Evaluation Script:** `python -m eval.evaluate_zenodo_rsvqa 5`

4. **VRSBench (`xiang709/VRSBench`)**
   - **Repository / Website:** [VRSBench Project Page](https://vrsbench.github.io/) | [Hugging Face Hub](https://huggingface.co/datasets/xiang709/VRSBench)
   - **Capabilities:** Detailed remote-sensing image captions, visual question-answering (VQA) pairs, and object grounding annotations.
   - **Evaluation Script:** `python -m eval.evaluate_vrsbench 5`

5. **RSVLM-QA (`StarZi0213/RSVLM-QA`)**
   - **Repository:** [RSVLM-QA GitHub](https://github.com/StarZi0213/RSVLM-QA)
   - **Capabilities:** Remote-Sensing Vision-Language Model question-answering benchmark for overhead scene understanding, land-use classification, and object counting.

## Running Benchmark Evaluation Scripts

Evaluate SatQuery AI's VQA, CDVQA, and Multimodal pipelines against all integrated benchmark datasets:
```bash
# 1. VRSBench Benchmark Evaluation
python -m eval.evaluate_vrsbench 5

# 2. BigEarthNet-MM Multimodal SAR-Optical Evaluation
python -m eval.evaluate_bigearthnet 5

# 3. CDVQA Bi-Temporal Change Reasoning Evaluation
python -m eval.evaluate_cdvqa 5

# 4. Zenodo RS VQA Count & Presence Benchmark Evaluation
python -m eval.evaluate_zenodo_rsvqa 5
```
