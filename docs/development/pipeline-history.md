# Pipeline stages and Git history

`main` is the integrated pipeline. The numbered `pipeline/*` branches mark
canonical commits already reachable from `main`. The numbers follow the
pipeline's logical order; the commits still have their original parent order.
Use `main` to run the current pipeline. Earlier branches capture the layout and
dependencies at each milestone, so they are reference points rather than
supported checkouts for a new run.

| Stage branch | Canonical tip | Author and committer | Contribution |
| --- | --- | --- | --- |
| `pipeline/01-cohort-audit` | [`89b41fd`](https://github.com/karmooPA/lung-wsi-mil/commit/89b41fde1c8cf888de5b6f9bc325d8be1b76ad5a) | `mragetsars` | Frozen TCGA-LUSC cohort, source provenance and repeatable dataset audit |
| `pipeline/02-preprocessing` | [`b439d92`](https://github.com/karmooPA/lung-wsi-mil/commit/b439d92eb96765f298194e87e6476d0334b3a9b0) | `karmooPA` | WSI reading, tissue detection, patch extraction, batch processing and QC |
| `pipeline/03-augmentation` | [`3d68dbb`](https://github.com/karmooPA/lung-wsi-mil/commit/3d68dbb24fee6dfb61e5dde02065334afa89526e) | `mragetsars` | Deterministic Original, Aug1 and Aug2 transforms |
| `pipeline/04-full-cohort` | [`e9c7d12`](https://github.com/karmooPA/lung-wsi-mil/commit/e9c7d12536003d9a17e2f901bdfab753ec109397) | `mragetsars` | Resumable full-cohort preprocessing, augmentation and independent verification |
| `pipeline/05-kimianet-encoder` | [`4cb5aee`](https://github.com/karmooPA/lung-wsi-mil/commit/4cb5aeef828d026f9a03a03a8feffc1e067f0a0d) | `karmooPA` | KimiaNet encoder and batched feature extraction with aligned coordinates |
| `pipeline/06-patient-splits` | [`0f16ca1`](https://github.com/karmooPA/lung-wsi-mil/commit/0f16ca1507d8aac57166253f6deeac2d18d6a243) | `karmooPA` | MIL data loading, patient-aware outer splits and frozen cross-validation folds |
| `pipeline/07-amil-model` | [`41133bb`](https://github.com/karmooPA/lung-wsi-mil/commit/41133bb7adba975118747634874aefa4a5cbcc28) | `karmooPA` | AMIL model, input checks and initialization |
| `pipeline/08-epoch-training` | [`98049d2`](https://github.com/karmooPA/lung-wsi-mil/commit/98049d2272013186928cb92afe0020f853476a62) | `karmooPA` | Padding and BCE checks, and batch/epoch training |
| `pipeline/09-verified-features` | [`2b86627`](https://github.com/karmooPA/lung-wsi-mil/commit/2b86627be0f04430e3e438fbef5d4264019e6c1f) | `mragetsars` | Official checkpoint verification, streaming feature cache and AMIL acceptance contracts |
| `pipeline/10-evaluation-metrics` | [`a2dc2d0`](https://github.com/karmooPA/lung-wsi-mil/commit/a2dc2d08041f0f7fbe009d65f76d88f589a5fa44) | `mragetsars` | Original-only inference, slide-level metrics and patient-cluster uncertainty |
| `pipeline/11-cross-validation` | [`2dc71e5`](https://github.com/karmooPA/lung-wsi-mil/commit/2dc71e5835d25e8f09b75a4b75585223e74b56c0) | `karmooPA` | Five-fold training foundation, validation-based checkpoint selection, early stopping and resume |
| `pipeline/12-test-and-heatmaps` | [`7b8f448`](https://github.com/karmooPA/lung-wsi-mil/commit/7b8f44843e80a3480fa63f0f882d560ee910b1f3) | `mragetsars` | Five-seed orchestration, final held-out tests, attention and spatial heatmaps |
| `pipeline/13-real-run` | [`7e8a2b0`](https://github.com/karmooPA/lung-wsi-mil/commit/7e8a2b03365978a08693857995f8809f4a5b7df0) | `mragetsars` | Launcher fix and exact source commit used for the completed server run |
| `pipeline/14-results` | [`57495fe`](https://github.com/karmooPA/lung-wsi-mil/commit/57495fe2325423284b4d8cbfcb66bf16d00a98e0) | `mragetsars` | Primary real-cohort report (`f5795ff`) and supplementary all-25 report (`57495fe`) |
| `pipeline/15-workspace-runner` | [784cf7e](https://github.com/karmooPA/lung-wsi-mil/commit/784cf7e870efe3cd24670bc7f29c1ef1873d0abb) | `mragetsars` | Portable pipeline command, workspace configuration and ordered workflow guide |

The author and committer column names the identity on the **tip commit**. A
branch also includes ancestors from earlier work by both contributors. See the
[ownership record](ownership.md), [original-to-canonical commit map](commit-map.csv)
and [five-fold attribution correction](roadmap-attribution-correction.json) for
the evidence behind the stage ownership.

## How the stages reached main

The existing history contains these merge commits:

| Merge | Work integrated |
| --- | --- |
| [`430b069`](https://github.com/karmooPA/lung-wsi-mil/commit/430b0698816e025a5749a772cf699695e9715aa4) | Frozen cohort audit (`89b41fd`) |
| [`7232b9b`](https://github.com/karmooPA/lung-wsi-mil/commit/7232b9be727e1c68bff165dd4eb992e34462d717) | WSI preprocessing ancestry (`b439d92`) and three-view augmentation (`3d68dbb`) |
| [`e4cdcb2`](https://github.com/karmooPA/lung-wsi-mil/commit/e4cdcb2c0f05c54c168a36381f63a6037c3d4422) | KimiaNet, MIL data infrastructure, patient splits and AMIL (`41133bb` and ancestors) |
| [`7c34da0`](https://github.com/karmooPA/lung-wsi-mil/commit/7c34da00c07ca333632af51764d0fbec7b67e6ed) | Verified full-cohort processing (`e9c7d12`) |
| [`802b78f`](https://github.com/karmooPA/lung-wsi-mil/commit/802b78ffb9b70f414bef2e0863d7fce67ddd3827) | Existing AMIL batch and epoch training (`98049d2`) |
| [f2ed877](https://github.com/karmooPA/lung-wsi-mil/commit/f2ed877c00295cbf5755a1223b82fab6a8e9691e) | Verified real-run launcher (stage 13) |
| [0bee5f1](https://github.com/karmooPA/lung-wsi-mil/commit/0bee5f17000df910ac63113ac1c8b4b9e94cd297) | Real-run and all-fold reports (stage 14) |
| [3c001eb](https://github.com/karmooPA/lung-wsi-mil/commit/3c001eb82ed064bcad06712d4de3d1e28c6b90d3) | Structured workspace runner (stage 15) |


Verified features, evaluation metrics, five-fold training and five-seed testing
entered main as ordinary commits. The verified launcher and the two real
result reports were integrated later through explicit merges. A stage branch adds a name
for its existing tip; it does not create a merge commit or duplicate the work.

The legacy `feature/amil-implementation` tip `4db691d` is a separate five-fold
API on the pre-organization history. It remains available as legacy work. The
integrated implementation is `2dc71e5`, with `karmooPA` as author and committer.
See [history and migration](git-history.md) for the earlier organization and
archive tags.

## Verify after fetching the branches

From the repository root, these read-only commands check that every published
stage tip is integrated into `origin/main` and that its reachable author,
committer and subject fields satisfy the repository policy:

```bash
set -e
python scripts/git_policy.py --revision origin/main
set -- $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/pipeline/)
test "$#" -eq 15
for stage do
  git merge-base --is-ancestor "$stage" origin/main
  printf '%s %s\n' "$stage" "$(git rev-parse --short "$stage")"
done
```

Compare the printed branches and tips with the table. To inspect the genuine
merge history or the identity recorded on an individual tip, use:

```bash
git log --first-parent --merges --oneline origin/main
git show -s --format=fuller origin/pipeline/11-cross-validation
```

Keep `pipeline/*` refs at their documented milestones. For new work, branch from
the current `main`, give each change its actual contributor identity, and
integrate after verification. Prefer a fast-forward for a linear series and
preserve genuine merge ancestry when branches have diverged. See
[Contributing](../../CONTRIBUTING.md) for branch and commit conventions.
