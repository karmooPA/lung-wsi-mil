# KimiaNet feature cache and AMIL acceptance procedure

The `main` branch connects the existing saved
Original/Aug1/Aug2 patch views to frozen KimiaNet features and the AMIL public
contract. This document describes the procedure and acceptance criteria.
Run-specific results, tested commits, hashes, counts and limitations are recorded
separately after execution; the presence of these commands is not evidence that
the real-data acceptance gate passed.

Batch/epoch training is available in `lung_wsi_mil.mil.training`; this gate does
not train a model. Checkpoint selection, ROC/AUC and heatmaps remain later work.
Randomly initialized AMIL predictions in this gate measure
interface and numerical correctness, rather than diagnostic performance.

## Environment

Use Python 3.12 and the `.[dev]` extra for this acceptance procedure, which also
runs pytest regressions. The `.[mil]` extra is sufficient for extraction and the
training APIs. Keep
PyTorch **2.14.0** and torchvision **0.29.0** as a pair. Install the selected
CPU/CUDA wheels first, then install the project extra; use a separate environment
to preserve the previous augmentation runtime.

CPU example on Ubuntu/WSL:

```bash
python3.12 -m venv .venv/lung-wsi-mil-cpu
.venv/lung-wsi-mil-cpu/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
.venv/lung-wsi-mil-cpu/bin/python -m pip install -e ".[dev]"
```

CUDA 13.0 example on the Windows server, whose driver must support that runtime:

```powershell
py -3.12 -m venv .venv\lung-wsi-mil
$milPython = '.\.venv\lung-wsi-mil\Scripts\python.exe'
& $milPython -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
& $milPython -m pip install -e ".[dev]"
& $milPython -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

For Windows CPU execution, create a separate CPU environment and replace the
wheel index with `https://download.pytorch.org/whl/cpu`. The extraction command
accepts `--device cpu` or `--device cuda`. The AMIL acceptance runner always
validates full real bags on CPU and also runs its small CUDA/MPS tests when those
devices are available.

## Genuine KimiaNet weights

The encoder follows the upstream
[KimiaLabMayo/KimiaNet repository](https://github.com/KimiaLabMayo/KimiaNet/tree/8a2f120999e04e4e18e0057bacbd1b47c27a56be)
at commit `8a2f120999e04e4e18e0057bacbd1b47c27a56be`, including its DenseNet feature
stack and adaptive average pooling. Use the upstream
[PyTorch checkpoint](https://raw.githubusercontent.com/KimiaLabMayo/KimiaNet/8a2f120999e04e4e18e0057bacbd1b47c27a56be/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth)
under its [End User Agreement](https://github.com/KimiaLabMayo/KimiaNet/blob/8a2f120999e04e4e18e0057bacbd1b47c27a56be/agreement.pdf).
The weights are for permitted academic, noncommercial research; do not
redistribute or transfer them to third parties. Obtain them from upstream and
keep them outside the source repository. They are not included in this project.

The pinned upstream checkpoint is 28,443,662 bytes with SHA256:

```text
f4972513e308d3197fb940177ca607bb9bb8f82690eadaec1e76190ef8b68375
```

Check the downloaded file against that identity before the real run. The loader
uses `torch.load(..., weights_only=True)`, converts official tensor names,
requires a complete matching state dictionary, rejects nonfinite tensors, and
freezes the model in evaluation mode. The cache records the actual weights
SHA256; it does not substitute random weights when loading fails.

## Frozen cohort and source contract

[`metadata/pilot14_selection.csv`](../../metadata/pilot14_selection.csv) pins the
same 14 slide IDs and patient identities used by the accepted preprocessing
pilot: seven Normal and seven Tumor slides. Labels and case IDs come from the
frozen [`metadata/dataset_inventory.csv`](../../metadata/dataset_inventory.csv).
See [the earlier preprocessing/augmentation result](../results/pilot14-2026-10-02.md)
for its per-slide identities and source hashes. That historical result does not
establish KimiaNet or AMIL acceptance.

The patch root can contain the full cohort. `--selection` restricts extraction
to the exact frozen 14 rather than choosing the first 14 files. Expected input:

```text
<patch-root>/slides/<slide_id>/augmentation/patch_views.h5
  images              uint8   [3,N,512,512,3]
  coords              int64   [N,2]
  tissue_percentage   float32 [N]
```

Existing augmentation metadata must agree with the inventory. Duplicate slide
IDs, missing sources, metadata mismatches and empty bags are explicit failures.
No selected slide is excluded to make the gate pass. Omitting `--selection`
requests all inventory slides and requires valid sources for every one.

## Extract features without loading a whole slide's images

From the repository directory on the Windows server, with weights already
obtained from upstream, run:

```powershell
$milPython = '.\.venv\lung-wsi-mil\Scripts\python.exe'
$patchRoot = 'D:\Data\TCGA_LUSC_HDF'
$weights = 'D:\Weights\KimiaNetPyTorchWeights.pth'
$featureCache = 'D:\Outputs\AMIL_Current\pilot14-kimianet.h5'

& $milPython scripts/extract_kimianet_features.py `
  --patch-root $patchRoot `
  --selection metadata/pilot14_selection.csv `
  --inventory metadata/dataset_inventory.csv `
  --weights $weights `
  --output $featureCache `
  --batch-size 8 `
  --device cuda
```

Use `--device cpu` for CPU extraction. The equivalent Ubuntu invocation uses
the CPU environment's `bin/python` and local source/weights/output paths.
Keep feature files and run reports outside Git, as in the example above.

The pipeline restores the existing ImageNet normalization with
`load_augmented_chunk`, holding only a bounded image chunk. Each view enters
the same frozen encoder separately. It preserves every patch and coordinate
and writes one slide group per selected WSI:

```text
<slide_id>/embeddings   float32 [3,N,1024]
<slide_id>/coords       int64   [N,2]
<slide_id>/label        scalar  0=Normal, 1=Tumor
<slide_id>/case_id      scalar  UTF-8 patient ID
```

The output includes source, weights, code, inventory, selection and
configuration fingerprints, runtime details and per-slide content hashes.
Every group must validate before the sibling staged file replaces the final
path atomically. A failed run cannot publish a partial feature cache.

An existing output is refused unless `--resume` is specified. Resume reuses
only a **complete** cache whose source/configuration/code/weights fingerprints
and every saved group remain valid. It does not resume individual feature
slides in an interrupted staged file. A caught failure removes its stage and
lock; an abrupt process kill can leave them behind. After establishing that no
writer remains, remove only those interrupted stage/lock artifacts and rerun
the extraction. Preserve a stale or corrupt completed output for diagnosis and
choose a new output path for regeneration.

## AMIL public contract and padding boundary

Defaults are `input_dim=1024`, `attention_dim=128`, one attention head and
`threshold=0.5`. Attention is Linear -> tanh -> Linear -> softmax across
instances. The bag representation is the attention-weighted sum of embeddings;
the classifier is Linear -> Sigmoid. The public forward return is exactly:

```python
Y_prob, Y_hat, A = model(x)
# x: [N,1024], N >= 1, floating and entirely finite
# Y_prob: [1,1], Y_hat: [1,1], A: [1,N]
# Y_hat = 1 when Y_prob >= threshold, including exactly 0.5
```

`initialize_weights(model, method="xavier")` initializes all Linear weights;
`method="kaiming"` is also supported. Linear biases are zeroed, and unsupported
methods fail clearly.

The existing Dataset, collate and DataLoader interfaces are preserved. Collate
expands `B` WSI to `3B` bags and pads embeddings with `-Inf` and coordinates
with `[-1,-1]`. AMIL consumes one clean finite bag rather than padded batches.
The padding helper in `tests/test_amil_pipeline.py` belongs only to validation;
it checks aligned complete tail sentinels before trimming. Production padding
removal lives in `lung_wsi_mil.mil.bag_utils` and is called by the training loop.

The public classifier returns probabilities. The integrated training loop uses
validated binary cross-entropy on those probabilities. `BCEWithLogitsLoss`
would require a separately agreed logits interface. See [training.md](training.md)
for the current optimizer API; the three-output model contract is preserved.

## Run the acceptance gate

The final runner includes AMIL unit/initialization tests, synthetic pipeline
tests, the genuine full Pilot14 forward pass, and seven existing data-layer
regression modules. Save its JSON report outside the repository:

```powershell
& $milPython scripts/validate_amil.py `
  --pilot14-h5 $featureCache `
  --output 'D:\Outputs\AMIL_Current\amil-validation.json'
```

For individual development tests on Ubuntu/WSL:

```bash
.venv/lung-wsi-mil-cpu/bin/python -m unittest discover -s tests -p 'test_amil*.py' -v
PILOT14_H5=../lung-wsi-mil-outputs/amil-contract/pilot14-kimianet.h5 \
  .venv/lung-wsi-mil-cpu/bin/python scripts/validate_amil.py \
  --output ../lung-wsi-mil-outputs/amil-contract/amil-validation.json
```

An unset `PILOT14_H5` may skip the real test during ordinary unit-test/CI runs.
That skip is **not acceptance**: `validate_amil.py` exits nonzero until the real
file passes. Unavailable optional CUDA/MPS devices produce explicit skips.

The real gate checks all of the following without patch subsampling:

- All 14 WSI load through `MILHDF5Dataset`, with finite nonempty `[3,N,1024]`
  embeddings, matching nonnegative coordinates, binary labels and readable IDs;
  the frozen cohort must retain its 7/7 label balance.
- A full `batch_size=4`, `shuffle=False` DataLoader pass yields exactly 42 views,
  with correct Original/Aug1/Aug2 order, repeated labels/case IDs and identical
  coordinates across each slide's views, including the final partial batch.
- Every batch uses its true maximum `N`. Padding is a contiguous tail with the
  exact embedding/coordinate sentinels, and trimming preserves all real rows.
- Every trimmed view equals its stored source embedding exactly after the
  existing Dataset's float32 conversion. Every one of the 42 full bags forwards
  through AMIL with the contracted shapes, finite probability/prediction,
  nonnegative finite attention, and attention sum approximately one.
- CPU unit tests cover independent attention/aggregation/classifier arithmetic,
  the inclusive threshold, permutation invariance, gradients, initialization,
  invalid inputs and padding rejection; existing data-layer regressions pass.

The JSON report records the full Git commit and branch, dirty status and tested
source hashes; OS/Python/PyTorch/device results; named pass/fail/skip outcomes;
the real HDF5 path/full SHA256 and stability check; WSI/view/forward counts;
minimum/maximum bag sizes; regression results and warnings. Inspect the report's
`acceptance` fields and process exit status before proceeding to training.
