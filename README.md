<<<<<<< HEAD
# SatQuery AI — Vision-Language Assistant for Remote Sensing

**Problem Statement ID:** SIH26167  
**Theme:** Space Technology | **Category:** Software | **Team:** SatVision  
**Hackathon:** Smart India Hackathon 2026

## Overview
SatQuery AI is an agentic vision-language assistant for satellite imagery. It allows users to upload optical or SAR remote-sensing images and query them using natural language. The system automatically routes queries to specialist analysis pipelines (VQA, Image Registration, Change Detection, SAR-Optical Fusion, Visual Grounding) and returns an explainable answer with confidence metrics and visual evidence.

## Project Structure
```
satquery-ai/
├── backend/            # FastAPI application & AI pipelines
├── ui/                 # Web interface (HTML/CSS/JS)
├── models/             # Pretrained model checkpoints & loaders
├── data/               # Datasets and preloaded demo samples
├── outputs/            # Generated masks, overlays, temporary outputs
├── tests/              # Pytest test suite
├── eval/               # Evaluation scripts
└── docs/               # Architecture, progress log, datasets documentation
```

## Quick Start (Windows)
```cmd
run.bat
```
Or start manually:
```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Then navigate to `http://localhost:8000` in your web browser.
=======
# SatQuery-AI
>>>>>>> 741dc2b3cb35fa6749248e1f8330cc6b0f33fc86
