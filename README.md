# Lung WSI MIL

MIL-based whole-slide image analysis for lung tumor detection.

## Current Task

Binary classification of lung pathology whole-slide images:

- Tumor
- Normal

## Planned Pipeline

WSI  
→ Tissue Detection  
→ Patch Extraction  
→ Pathology Encoder  
→ Patch Embeddings  
→ Attention MIL  
→ Tumor / Normal Classification  
→ Attention Heatmap

## Team

- Mohammad
- Meraj

## Status

Project initialization.

## Dataset audit

See [docs/dataset-audit.md](docs/dataset-audit.md) for the reproducible audit,
source provenance, patient grouping and remaining full-data checks. Run
`python scripts/inspect_dataset.py` to replay the recorded pilot evidence
without requiring WSI files. This is not a fresh full-dataset scan.
