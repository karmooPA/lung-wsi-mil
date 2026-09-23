# Patch augmentation

Base: `feature/wsi-preprocessing` at
`a3f699f89beb7c7ed958f596877765abbe326ba2`, not main.
This branch adds augmentation without changing the colleague's preprocessing,
feature extraction, shared requirements or manual demo.

## Contract and use

```python
import numpy as np
from src.preprocessing.patch_extraction import extract_patch
from src.augmentation import transform_patch, transform_batch

rng = np.random.default_rng(42)  # create once per run/worker, then reuse
patch = extract_patch(slide, level0_x=4096, level0_y=4096,
                      target_downsample=4.0, patch_size=512)
# Obtain target_downsample from preprocess_slide(...)["resolution"], not a
# universal assumption of 4.0: it depends on the scanned slide's resolution.
original, aug1, aug2 = transform_patch(patch, rng=rng)
batch = transform_batch([patch], rng=rng)
assert batch.shape == (3, 1, 3, 512, 512)
```

Input must already be `PIL.Image.Image`, RGB, exactly 512x512, as returned by
the existing `extract_patch`. Wrong modes/sizes/types raise an error; no hidden
resize, colour conversion, crop or normalization occurs at the API boundary.
Each output is CPU `torch.float32`, normalized once using ImageNet means
`(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`.
Original has no random transform. The API does not mutate the input image.

Batch axes: **[representation, patch index, channel, height, width]**.
Representations are always Original, Aug1, Aug2. Patch order is retained;
rotating a patch's contents never changes its WSI origin coordinate. Empty input
returns `[3, 0, 3, 512, 512]`.

Each augmented view independently runs H/E perturbation -> Gaussian blur ->
quarter-turn rotation -> horizontal flip -> vertical flip -> normalization.
Both views start from the same original patch. The two Compose instances use
fresh draws from an advancing NumPy Generator; they do not reuse a parameter
set or augment Aug1 to create Aug2. Independent draws may coincide, especially
on a blank/symmetric patch. Artificially forcing inequality would bias sampling.

Pass one explicit generator across a batch or series of calls for reproducibility.
Calling single-patch transforms in order with the same initial generator state
matches transform_batch exactly. With `rng=None`, module-level pipelines use
an entropy-seeded stream. `np.random.seed` and `torch.manual_seed` do not seed
that stream. Parallel workers must each construct their own generator, e.g.
using `np.random.SeedSequence(run_seed).spawn(worker_count)`; never reuse a
forked generator's state or recreate the same seed for every patch. Reproducible
outputs also depend on library versions and input order. Global RNG states are
not modified by the augmentation code.

## Installation and tests

Tested with Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0, scikit-image 0.26.0,
PyTorch 2.14.0 and torchvision 0.29.0, using CPU execution. Use Python 3.12
for the pinned dependency set. `requirements-augmentation.txt` includes the
existing OpenSlide/Pillow requirements, required array/image/torch libraries,
and h5py 3.16.0 for the optional KimiaNet integration test.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-augmentation.txt
python -m unittest discover -s tests -p 'test_augmentation*.py' -v
```

For a CPU-only environment, install the matching torch/torchvision pair from
the official CPU index first, then the requirements file. For CUDA use a
compatible official PyTorch wheel pair for the intended hardware. The tested
environment used the default +cu130 wheels on CPU; GPU execution was not tested.

```bash
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-augmentation.txt
```

On this branch, 14 core tests pass and 4 KimiaNet tests skip because those modules
are absent. On a temporary checkout with KimiaNet `b142bc2`, all 18 tests run:
numeric normalization parity, axis/order preservation, RNG independence and
repeatability, H/E/D handling, blur semantics, exact rotations/flips, finite
black/white outputs, strict inputs, empty/singleton batches, the actual
`extract_patch` with a synthetic slide, QC export, actual feature batching and
HDF5 writing, and an untrained KimiaNet forward pass. The known-output stand-in
encoder checks batching/coordinates; it is not a substitute for trained weights.

Use the narrow test pattern shown above. Existing `tests/test_preprocessing.py`
is a manual script with hard-coded `/Users/mohammad/...` WSI/output paths and
side effects on import. It is neither a portable unit test nor run by this
suite; this branch leaves it unchanged.

## Scientific and implementation decisions

| Source requirement / ambiguity | Implementation and reason |
| --- | --- |
| Task PDF pp. 2-3: ToTensor + ImageNet normalization | Same operations; public API enforces float32. Original matches `src/features/transforms.py` bit for bit in the tested environment. |
| PDF p. 3: independent H/E factors N(1, 0.02) | Scalar factors per image; D retained; scikit-image RGB -> HED -> RGB; clip to [0,1], quantize to uint8. No Macenko/stain normalization is added. |
| Article p. 5 says Gaussian noise addition; PDF p. 4 calls GaussianBlur | Follow the executable brief: blur. `RandomGaussianBlur` has the accurate name; `RandomGaussianNoise` remains a documented compatibility subclass. No additive noise is silently substituted. |
| PDF passes a length-one ndarray to GaussianBlur | Convert the sampled radius to a Python float. The literal ndarray call fails in tested Pillow 12.3. |
| Radius N(0, 0.5) may be negative | Use abs(radius), a half-normal magnitude, not clipping negative draws to zero. Tested Pillow scalar +/-0.5 produce equal blurred pixels; absolute value makes non-negative sigma explicit. |
| PDF rotation uses ndarray of quarter turns | Use a scalar integer and exact PIL transpose; no interpolation/crop for square patches. |
| Horizontal/vertical flip probability 0.5 | Independent draws using NumPy Generator, instead of mixing PyTorch and NumPy global RNGs. Distribution matches; bitwise upstream seed replay is not claimed. |
| Two views and `[3,N,3,512,512]` | Two fresh transform runs; same patch/coordinate index across representations. |

Scientific reference: supplied Afonso et al., *Multiple Instance Learning for
WSI: A comparative analysis of attention-based approaches*, JPI 15 (2024)
100403, p. 5; DOI `10.1016/j.jpi.2024.100403`. Pipeline-selection PDF provides
the project context, not additional executable parameters.
The authors' repository is [WSI_MIL_ROI](https://github.com/timafonso/WSI_MIL_ROI).
Its landing page was accessible; the implementation files could not be retrieved
in this session. Therefore the detailed operator contract is attributed to the
supplied task PDF, not an independently verified upstream revision. This is an
implementation of the brief with documented corrections, **not exact paper
reproduction**. Blur versus additive noise remains a scientific protocol choice
for team review; a noise protocol would need a separately specified pixel-scale
variance and new validation.

## KimiaNet connection and remaining caller work

Checked alongside `feature/kimianet-features` commit
`b142bc2af3827955434e0eff9d74d31c19cccffc` in a temporary checkout. That branch
is a direct descendant of this preprocessing base with two feature commits.
No colleague feature code is copied into this patch or altered for the test.

`extract_embeddings` already accepts `[3,N,3,512,512]` and returns `[3,N,1024]`.
`process_transformed_slide` already writes the aligned embeddings, coordinates,
slide/case IDs and label. No API change is needed. The missing production piece
is the caller connecting the existing stages, with a loaded, frozen, eval-mode
model and real WSIs. In a combined checkout, the handoff is:

```python
import numpy as np
from src.augmentation import transform_batch
from src.features.pipeline import process_transformed_slide

# patches: list of actual extract_patch outputs, in coordinates order
# coordinates: level-zero (x,y) for those patches, not thumbnail coordinates
transformed = transform_batch(patches, rng=rng)
coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 2)
saved = process_transformed_slide(
    transformed, coords, model, device, hdf5_path,
    slide_id, patient_id, label, batch_size=16,
)
```

Do not apply `kimianet_transform` a second time to normalized tensors. Use the
TCGA slide barcode as slide_id and the patient grouping ID as case_id; encode
Normal/Tumor as 0/1 consistently with the supplied labels. For N=0 the existing
coordinate validator needs shape `(0,2)`; a bare `[]` fails. The example handles
that without changing the colleague's implementation. `load_kimianet_weights`
already freezes the model and calls eval; actual weights were not available.

The full image tensor alone costs **9 MiB per patch**, roughly 288 MiB at N=32.
The colleague's `batch_size` limits model inference, not allocation of this
entire tensor. For large slides, the caller/writer needs a separate streaming
extension: transform/extract chunks in coordinate order and append into the same
HDF5 slide datasets. Do not call the current writer repeatedly with the same
slide_id for successive chunks: it returns False after the first write and
would drop later chunks. This production streaming work is outside the required
augmentation API and remains an explicit integration task for the feature owner.

## Visual QC and remaining data gate

Neither raw WSIs nor `tumor_patch_4096_4096.png` was supplied. The two images in
the audit package are whole-slide thumbnails, not the requested extracted patch;
they were not resized/cropped and presented as that patch. The software QC
panel was generated and visually inspected using a clearly synthetic colour
pattern: three panels, correct orientation, visible stain-space changes and
intact image boundaries. Arbitrary RGB patterns are outside the H&E colour
model, so that exercise cannot establish preservation of tissue morphology.

On receipt of the actual 512x512 RGB patch, run:

```bash
python scripts/preview_augmentation.py /path/to/tumor_patch_4096_4096.png --seed 42 --output results/augmentation/real-patch-qc.png
```

The script verifies shape/dtype/finiteness, denormalizes the three tensors,
exports a PNG panel and JSON containing the input/output hashes, seed and
versions. Review stain changes, rotations/flips, tissue structure, edge integrity,
clipping and blur. Also exercise the real extract_patch -> augmentation ->
trained KimiaNet path with the intended WSI, magnification and weights. Those
real-data checks remain open; the PDF's complete Definition of Done is not met.
All views of every patient's slides must remain in the same future split.
