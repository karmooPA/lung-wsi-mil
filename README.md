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

Implemented and tested:

- reproducible TCGA-LUSC metadata audit;
- WSI tissue detection and Level-0 patch-coordinate extraction at a requested magnification;
- strict patch QC with no arbitrary minimum patch-count exclusion;
- deterministic Original/Aug1/Aug2 augmentation;
- lossless, chunk-readable HDF5 storage for a balanced 14-WSI pilot;
- full-dataset orchestration with bounded slide workers and per-slide resume;
- independent artifact verification, source re-extraction checks, resume validation and QC montages.

KimiaNet feature extraction exists on the separate `feature/kimianet-features`
branch and is intentionally not part of `main` yet. The current main-branch
pipeline ends at augmentation storage.

## Environment

The supported development environment is Ubuntu under WSL with Python 3.12.
Raw WSIs remain outside Git and are read from the mounted dataset drive.

The local WSL workspace uses three sibling directories:

```text
/home/mrage/Codes/Python/
├── lung-wsi-mil/          # Git repository: code, tests, metadata, documentation
├── lung-wsi-mil-inputs/
│   └── pilot14/           # The 14 copied WSIs, verified against inventory MD5
└── lung-wsi-mil-outputs/
    ├── pilot14-final/     # Current accepted preprocessing/augmentation artifacts
    ├── quality/           # Test and audit reports
    └── archive/           # Earlier runs, preserved as historical evidence
```

Drive M is the original data source. The pilot uses the verified local copies
so processing is independent of drive disconnections. Directory layout and
reproduction commands are documented in [docs/workspace-layout.md](docs/workspace-layout.md).

```bash
python3.12 -m venv .venv/lung-wsi-augmentation
.venv/lung-wsi-augmentation/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
.venv/lung-wsi-augmentation/bin/python -m pip install -r requirements-augmentation.txt
```

Run every portable compile, unit, integration and recorded-audit check:

```bash
.venv/lung-wsi-augmentation/bin/python scripts/run_quality_checks.py
```

Run and independently verify the balanced real-data pilot:

```bash
bash scripts/run_pilot14.sh

.venv/lung-wsi-augmentation/bin/python scripts/verify_pilot14.py \
  ../lung-wsi-mil-outputs/pilot14-final \
  --verify-source-md5 --source-sample-count 100000
```

The runner selects seven Normal and seven Tumor slides from distinct patients,
writes atomically staged per-slide outputs, and refuses to resume stale or
corrupt artifacts. See [docs/pilot-14-wsi.md](docs/pilot-14-wsi.md) and the
[verified pilot result](docs/pilot-14-wsi-results.md).

## Full dataset

Run `scripts/run_wsi_dataset.py` to process every inventory-resolved local WSI,
then use `scripts/verify_wsi_dataset.py` for independent artifact checks, source
MD5 validation and Original patch re-extraction. The full-dataset runner allows
multiple slides per patient and preserves the pilot's Original/Aug1/Aug2 HDF5
format. Patient-level splitting is still required for later model evaluation.

See [docs/full-dataset.md](docs/full-dataset.md) for Windows and generic commands,
six-worker execution, progress files and interruption/resume recovery.

## Dataset audit

See [docs/dataset-audit.md](docs/dataset-audit.md) for the reproducible audit,
source provenance, patient grouping and remaining full-data checks. Run
`python scripts/inspect_dataset.py` to replay the recorded pilot evidence
without requiring WSI files. This is not a fresh full-dataset scan.
