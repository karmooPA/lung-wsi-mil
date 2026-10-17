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

The `feature/amil-contract-validation` branch integrates KimiaNet extraction,
the MIL data layer and AMIL. It adds a bounded streaming feature-cache CLI using
genuine frozen KimiaNet weights, safe strict checkpoint loading, and AMIL unit,
initialization and Dataset/DataLoader integration tests. The final acceptance
runner requires all 14 frozen pilot slides and all 42 full view-bag forwards;
an ordinary CI skip does not satisfy that gate. Run-specific acceptance evidence
is recorded separately from this procedure. The
[2026-10-17 server acceptance result](docs/amil-validation-results-2026-10-17.md)
records the genuine 14-slide/42-bag PASS and its source/artifact identities.

Training, optimization, ROC/AUC evaluation and attention heatmaps remain later
work. AMIL currently returns exactly probability, binary prediction and attention
weights; its padding-removal helper remains test-only.

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

## KimiaNet and AMIL

Install `requirements-mil.txt` in a separate Python 3.12 environment, keeping
`torch==2.14.0` paired with `torchvision==0.29.0`. The
[feature-cache and AMIL validation procedure](docs/amil-validation.md) contains
CPU and CUDA 13.0 installation commands, upstream KimiaNet checkpoint identity
and terms, and the complete public model/padding contract. Obtain weights from
upstream and retain them outside Git; this repository does not redistribute them.

From the Windows server repository directory, after installing the CUDA
environment and obtaining the genuine checkpoint:

```powershell
$milPython = '.\.venv\lung-wsi-mil\Scripts\python.exe'
$featureCache = 'D:\Meraj Rastegar\Outputs\amil-contract\pilot14-kimianet.h5'
& $milPython scripts/extract_kimianet_features.py `
  --patch-root 'D:\Meraj Rastegar\Data\TCGA_LUSC_HDF' `
  --selection metadata/pilot14_selection.csv `
  --inventory metadata/dataset_inventory.csv `
  --weights 'D:\Meraj Rastegar\Weights\KimiaNetPyTorchWeights.pth' `
  --output $featureCache --batch-size 8 --device cuda

& $milPython scripts/validate_amil.py `
  --pilot14-h5 $featureCache `
  --output 'D:\Meraj Rastegar\Outputs\amil-contract\amil-validation.json'
```

Use `--device cpu` with a CPU environment. Extraction `--resume` reuses a
complete cache only after validating matching provenance and every group;
interrupted feature stages rerun. The acceptance command exits nonzero for a
missing real HDF5, any required failure/skip, or incomplete 14-WSI/42-view
coverage. Keep generated caches and JSON reports outside the source repository.

## Dataset audit

See [docs/dataset-audit.md](docs/dataset-audit.md) for the reproducible audit,
source provenance, patient grouping and remaining full-data checks. Run
`python scripts/inspect_dataset.py` to replay the recorded pilot evidence
without requiring WSI files. This is not a fresh full-dataset scan.
