#!/usr/bin/env python3
"""Verify the frozen cohort against its GDC snapshot without network access."""
import json
import argparse
import random
from collections import Counter
from pathlib import Path
from inspect_dataset import SOURCE, DEFAULT_OUTPUT_ROOT, load_expected, hash_file


def validate(source=SOURCE):
    source = Path(source)
    rows = load_expected(source/'labels_TCGA_LUSC_694.csv', source/'gdc_manifest_TCGA_LUSC_694.txt', source/'gdc_metadata_snapshot.json')
    snapshot = json.loads((source/'gdc_metadata_snapshot.json').read_text())
    hits = snapshot['data']['hits']
    normal = sorted((r for r in hits if r['cases'][0]['samples'][0]['tissue_type'] == 'Normal'), key=lambda r:r['file_id'])
    tumor = sorted((r for r in hits if r['cases'][0]['samples'][0]['tissue_type'] == 'Tumor'), key=lambda r:r['file_id'])
    selected = random.Random(42).sample(tumor, 347)
    expected_ids = {r['file_id'] for r in normal+selected}
    actual_ids = {r['file_id'] for r in rows}
    checks = {
        'snapshot_count': len(hits) == 1100,
        'candidate_normal_count': len(normal) == 347,
        'candidate_tumor_count': len(tumor) == 753,
        'cohort_count': len(rows) == 694,
        'balanced_labels': dict(Counter(r['label'] for r in rows)) == {'Normal':347,'Tumor':347},
        'all_label_encodings_consistent': all(r['label_valid'] for r in rows),
        'patient_barcode_agreement': all(r['patient_id'] == r['slide_id'][:12] for r in rows),
        'seed_42_selection_reproduced': expected_ids == actual_ids,
        'all_selected_rows_snapshot_verified': all(r['snapshot_verified'] for r in rows),
        'unique_slide_ids': len({r['slide_id'] for r in rows}) == len(rows),
        'unique_filenames': len({r['file_name'].casefold() for r in rows}) == len(rows),
        'unique_expected_md5': len({r['expected_md5'] for r in rows}) == len(rows),
    }
    result = {'status':'PASSED' if all(checks.values()) else 'FAILED', 'checks': checks,
              'snapshot_records':len(hits), 'cohort_records':len(rows),
              'selection_algorithm':'random.Random(42).sample(sorted(tumor, key=file_id), 347)',
              'source_hashes_sha256':{p.name:hash_file(p) for p in sorted(source.iterdir()) if p.is_file()}}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_ROOT/'dataset-audit/source_validation.json')
    args = parser.parse_args()
    result=validate(args.source)
    path=args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASSED' else 1)
