# AMIL batch and epoch training

The integrated `mil` package contains the training work originally committed by
karmooPA on `feature/amil-implementation`. It optimizes one finite view-bag at a
time using the existing three-return AMIL API.

```python
from torch.optim import Adam

from lung_wsi_mil.data.mil_dataloader import create_mil_dataloader
from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset
from lung_wsi_mil.mil.amil import AMIL
from lung_wsi_mil.mil.training import train_one_epoch

model = AMIL()  # float32 on CPU; move it before creating/using the optimizer
optimizer = Adam(model.parameters(), lr=1e-4)
dataset = MILHDF5Dataset("/external/features.h5")
# For an experiment, subset this dataset using a frozen patient-aware train split.
loader = create_mil_dataloader(dataset, batch_size=2, shuffle=False)
metrics = train_one_epoch(model, loader, optimizer)
print(metrics)
```

This is an API example, not an experiment protocol or selected hyperparameter
configuration. Use only the training partition when fitting a model.

`train_batch` validates the six-item collate tuple, removes complete `-Inf`
padding rows, computes one probability per view-bag, then takes one optimizer
step on mean binary cross-entropy. `remove_padding` rejects empty/all-padding
bags and nonfinite values in non-padding rows. AMIL itself does not remove padding.

`train_one_epoch` reports bag-weighted mean loss, batch count, view-bag count,
and **distinct** slide count across the entire epoch. Repeated views or repeated
sampling of one slide do not inflate `num_slides`. `device=None` uses the model's
device; `device="cuda"` resolves to its CUDA index. An explicitly different
indexed device is rejected. Keep model and input dtypes compatible.

The real AMIL + variable-length HDF5/DataLoader integration test verifies an
optimizer update. It uses synthetic embeddings and does not measure clinical
classification performance. Validation, early stopping, checkpoint selection,
five-fold/five-seed training, held-out metrics and attention/heatmaps are now
implemented in the [complete experiment workflow](experiments.md).
