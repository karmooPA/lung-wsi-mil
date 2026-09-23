# Dataset audit

Based on `main` at `6f18b2ecd4822416fbfc79f31f56812f35c10a4f`.
The audit brief's full-image Definition of Done remains open. No raw WSI is
included, downloaded or changed by this branch.

## Verified delivery state

| Item | Result and scope |
| --- | --- |
| Frozen cohort | 694 distinct slide IDs / filenames / GDC file UUIDs |
| Slide labels | 347 Normal, 347 Tumor |
| Patients | 373; 233 with Normal, 293 with Tumor, 153 in both groups |
| Slides per patient | 174 have 1; 98 have 2; 80 have 3; 21 have 4 |
| Declared size | 109,199,574,365 bytes (101.7000 GiB); all `.svs` |
| Prior image evidence | 2 OpenSlide probe + MD5 successes recorded in the supplied archive |
| Other images | 692 untested; geometry and readability `NA` |
| Fresh WSI inspections in this delivery | 0; no raw images available |
| Current presence/absence | Unknown for all 694 files |
| Duplicate scope | No duplicate logical identities or declared MD5s; full local hashing pending |

The two prior observations are **imported evidence**, not independently repeated
image measurements. Both report 20x objective power, MPP X/Y 0.5015 and two levels.
The Tumor observation is `50031276-21a2-4478-bc83-53a1abf6593a` (8000x9269),
and Normal is `734286d8-e7fa-4788-976c-3d3b591d3a7e` (12000x8785).
Neither was a random quality sample. These values say nothing about the other images.

## Provenance and selective integration

`metadata/source/` preserves the supplied manifest, labels, 1,100-record GDC
snapshot, selection record and pilot manifest byte for byte. These came from
`Lung_WSI_Dataset_Audit_Project(2).zip`, SHA256
`54abb7bf1b5f667dbc2558f8e75290cf9c9effc04ffe242790b60764bfd23e0b`.
The original archive's recorded acquisition date/release is a source claim;
this work cross-checks the frozen records offline, not the current GDC API.

`pilot_observations.json` contains only the archive's two successful inventory
rows, the original inventory/archive hashes, input hashes, runtime versions and
the report's UTC timestamp. The time is the historical run timestamp, not an
individual acquisition time. Import validates source hashes, IDs, labels,
size/checksum agreement and geometry. It cannot authenticate missing WSI bytes.

Selection is independently reproduced: retain 347 Normal records, sort the
753 Tumor candidates by UUID, and select 347 using Python `random.Random(42)`.
`scripts/validate_sources.py` verifies all 694 selected rows against the snapshot.
The article does not specify its tumor UUIDs/seed: equal cohort size does **not**
establish an identical research cohort or exact reproduction.

The article's abstract (p. 1) and supplied pipeline-selection PDF p. 3 say 349
slides per class; the article's TCGA-LUSC section (p. 4) and this frozen cohort
support **347 per class, 694 total**. This is an inconsistency within the paper
itself, repeated in the secondary document. The manifest is not padded to 698.
Source CRLF bytes are intentionally preserved with `metadata/.gitattributes`;
apply the mailbox patch using `git am --keep-cr` as in the delivery guide.

The scanner and regression tests were adapted from the supplied standalone
package. No preprocessing or feature code was replaced. Omitted: PDF generator,
fonts, thumbnails, historical test logs, auto-installing launcher and duplicate
README files. CSV/JSON plus this Markdown report are the reviewable outputs.
The full snapshot is retained because it makes the 1,100-to-694 selection
independently reproducible.

## Run

Python 3.10+; no third-party dependency is needed for offline runs or audit tests.
Run from the repository root. Outputs default to ignored `results/dataset-audit/`
so a routine run does not overwrite the committed handoff.

```bash
python scripts/validate_sources.py
python scripts/inspect_dataset.py
python -m unittest discover -s tests -p 'test_audit.py' -v
# Explicitly regenerate the committed evidence inventory:
python scripts/inspect_dataset.py --mode recorded --output-dir metadata
python scripts/validate_sources.py --output metadata/source_validation.json
```

Inventory and patient/issue tables reproduce byte for byte in recorded mode.
Summary generation time and installed-software fields vary by run. Local image
measurement timestamps also vary. Inputs, sorting and file measurements determine
all other values; nothing is derived from PDF example numbers.

For new image measurements, install only the audit requirements:

```bash
python -m pip install -r requirements-audit.txt
# Complete dataset root; expected-but-absent files are missing:
python scripts/inspect_dataset.py --mode local --data-root /path/to/TCGA_LUSC_WSI --verify-md5 --strict
# A deliberately small sample; unobserved cohort members stay untested:
python scripts/inspect_dataset.py --mode evidence --data-root /path/to/pilot --verify-md5 --output-dir results/pilot
# Ignore all historical image evidence and inspect only source metadata:
python scripts/inspect_dataset.py --mode metadata --output-dir results/cohort-only
```

On Windows, use 64-bit Python and a quoted path, e.g. `--data-root "D:/TCGA_LUSC_WSI"`.
No data path is hard-coded. Local mode requires an existing, explicitly provided
root. `--verify-md5` reads every present, size-matching file in full; a full pass
reads approximately 109.2 GB. Without it, readiness is explicitly size/probe only.
Do not run simultaneous scans into one output directory.

Exit codes: `0` means reports written (not necessarily ready); `2` means
invalid inputs/setup/dependency failure; `3` with `--strict` means checks did
not pass. Recorded/metadata/evidence modes can never pass strict readiness.

## Output contract

The first 14 inventory columns are those requested in the brief: slide_id,
patient_id, label, relative_path, file_extension, file_size_mb, width, height,
level_count, mpp_x, mpp_y, objective_power, readable, duplicate.

| Fields | Meaning |
| --- | --- |
| slide_id / file_id | Official TCGA slide barcode / GDC file UUID (different IDs) |
| patient_id | GDC case submitter ID, checked against the TCGA patient barcode |
| label / label_valid / label_error | Metadata-derived label and conflicts; never inferred from appearance |
| relative_path / path_source | POSIX path relative to the supplied root; actual path when found, expected UUID/filename otherwise |
| expected_relative_path | Manifest layout suggestion; not proof of a file's existence |
| file_size_bytes / file_size_mb / file_size_source | Measured, historically measured or manifest size; MB = 1,000,000 bytes |
| expected_file_size_bytes / expected_md5 | Frozen manifest values |
| size_matches / computed_md5 / md5_matches | Current or recorded integrity checks, disambiguated by observation_source |
| width / height / level_count / level_dimensions / level_downsamples | Native level-zero geometry and pyramid, never inferred from target magnification |
| mpp_x / mpp_y / objective_power | Positive finite native metadata; no guessed magnification |
| readable / probe_regions_read | OpenSlide header plus up to five 64x64 corner/centre probes per level |
| metadata_opened / missing_wsi_metadata | Header opened and observed missing fields; failure to open is not evidence of missing MPP |
| observation_source / observation_time_utc | `current_run`, `imported_pilot`, or `NA`; timestamp scope above |
| duplicate / duplicate_reasons | All matching slide ID, filename, path, UUID, declared or computed MD5 groups retained |
| patient_mixed_labels | Both Normal and Tumor tissue records exist for this patient; not automatically a label error |
| snapshot_verified | Identity/source cross-check, not verification of local image bytes |

CSV `True` / `False` / `NA` must be parsed explicitly; `bool("False")` is wrong.

| Situation | present | readable | status |
| --- | --- | --- | --- |
| No observation | NA | NA | not_inspected |
| Imported pilot success | NA | True | recorded_inspected |
| Absent from a complete local scan | False | NA | missing |
| Not located during a scan with errors/skipped links | NA | NA | not_located_in_incomplete_scan |
| Present, size mismatch (probe skipped) | True | NA | size_mismatch |
| Attempted OpenSlide probe fails | True | False | unreadable |
| Present, probes pass | True | True | inspected |
| Checksum mismatch / file I/O failure / changing file | See fields | See fields | checksum_mismatch / file_io_error / changed_during_scan |

An unreadable file needs investigation: unsupported format or I/O errors can
also cause a failed read. Neither `unreadable` nor size mismatch alone proves
corruption. A hash failure does not overwrite a successful readability probe.
Symbolic links are skipped and reported; a scan with errors cannot establish
absence. Discovery handles extensions case-insensitively; MRXS companions must
remain with their index. Unknown files are retained without invented labels.

`dataset_summary.json` separates current versus imported inspections, unknown
presence, readability, expected/observed sizes, missing fields among opened
headers, and readiness. `patient_summary.csv` counts each expected slide once;
`mixed_label_patients.csv` lists the 153 shared patients. `duplicates.csv`,
`issues.csv`, `partial_files.csv` and the runtime `audit.log` support follow-up.
Header-only issue tables do not imply full image verification.

## Leakage prevention and remaining gate

All slides and all derived patches/views/embeddings from a given `patient_id`
must stay in **one** train/validation/test partition. Split the 373 patients,
then map slides to partitions; never split the 694 rows independently. Retain
both tissue labels for mixed patients and check patient-set intersections are
empty. `Normal` is Solid Tissue Normal from this cancer cohort, not a claim of
healthy-donor status. No actual split or downstream model is implemented here.

To finish the brief: run full local mode on all 694 files after downloads finish,
review missing/partial/extra/duplicate/integrity/metadata problems, and record
exclusions with the team. Review regenerated reports before committing them.
Even a complete probe+MD5 pass is not exhaustive pixel or histology QC. Fresh
verification of the two pilot WSIs is also pending in this environment. Push,
PR creation and main merge are deliberately left to the user/reviewer.
