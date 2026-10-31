#!/usr/bin/env bash
# Run the pinned cohort using separate sibling input/output directories.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_parent="$(dirname -- "$project_root")"
python_path="${LUNG_WSI_PYTHON:-$project_root/.venv/bin/python}"
input_dir="${LUNG_WSI_INPUT_DIR:-$workspace_parent/$(basename -- "$project_root")-inputs/pilot14}"
output_dir="${LUNG_WSI_OUTPUT_DIR:-$workspace_parent/$(basename -- "$project_root")-outputs/pilot14-final}"

if [[ ! -x "$python_path" ]]; then
    printf 'Python environment not found: %s\n' "$python_path" >&2
    exit 2
fi

exec "$python_path" "$project_root/scripts/run_pilot14.py" \
    "$input_dir" "$output_dir" \
    --inventory "$project_root/metadata/dataset_inventory.csv" \
    --selection-csv "$project_root/metadata/pilot14_selection.csv" \
    --seed 42 --chunk-size 8 --qc-sample-count 6 --verify-source-md5 \
    "$@"
