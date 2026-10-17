# Real KimiaNet / AMIL acceptance — 2026-10-17

**PASS.** Genuine upstream KimiaNet weights produced features for all 14 frozen
pilot slides, and all 42 full view-bags passed AMIL validation on the server.
The final acceptance report completed at `2026-10-17T11:34:21.852138+00:00`;
the full quality runner completed at `2026-10-17T11:35:14.977833+00:00`.
This is numerical/pipeline acceptance, with no model-training or AUC claim.

The tested code was the clean commit
`c4ce4a171ba7fde782ef8788f4e4076e90c38ee1` on
`feature/amil-contract-validation`. Documentation committed afterward does not
change which code produced these results. The preserved full-dataset processing
branch is `feature/full-dataset-processing` at
`cf42b2bf648d73c035ec1bdf850b9c3cc84d61c9`.

For repository synchronization, the server's HTTPS Git Credential Manager
could not access/persist credentials in the SSH session. GitHub fetch/push used
the existing local SSH identity, with GitHub SSH over port 443 when port 22
stalled; the server received a Git bundle of those verified commits over its
authenticated SSH connection. No private key or token was copied to the server.
This preserved the exact history: the tested clean server tree matched pushed
commit `c4ce4a171ba7fde782ef8788f4e4076e90c38ee1`; a direct authenticated server
`git pull` is not claimed.

## What ran now, and what already existed

The full **694-slide preprocessing/augmentation run was performed earlier, on
2026-10-08**, with its previous verification recorded as PASSED. The current
work reconciled that existing cohort with the frozen metadata: exactly 694
selected slides, 347 Normal and 347 Tumor, 35,366 patches, and no missing or
unexpected slide IDs, identity errors or zero-patch slides in the reconciliation.
It did not rerun the entire historical raw-WSI MD5/re-extraction verification.
The quality runner's recorded-audit replay and frozen-source validation are
metadata/evidence checks, rather than a new full raw-WSI scan.

**The current KimiaNet feature extraction covered the frozen 14 only.** It read
their existing patch HDF5 files under
`D:\Meraj Rastegar\Data\TCGA_LUSC_HDF`, selected by
[`metadata/pilot14_selection.csv`](../metadata/pilot14_selection.csv). These
are the same seven Normal and seven Tumor slides described in the
[earlier pilot preprocessing result](pilot-14-wsi-results.md). No full-cohort
feature cache or training run is claimed here.

## Environment and measured extraction invocation

| Item | Recorded value |
| --- | --- |
| Execution host | Windows 11, build 26200, AMD64 |
| Repository | `D:\Meraj Rastegar\Source` |
| Runtime | Python 3.12.14 |
| PyTorch / torchvision | `2.14.0+cu130` / `0.29.0+cu130` |
| Extraction GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| Feature extraction | CUDA, frozen encoder in evaluation mode, batch size 8 |
| AMIL full real-bag validation | CPU, DataLoader batch size 4, `shuffle=False` |
| Optional devices | CUDA forward/backward PASS; MPS unavailable and skipped |
| One extraction invocation | 55.3891192 seconds, exit code 0 |

The extraction duration is an observed wall-clock invocation, including its
pipeline work. It is not a controlled throughput benchmark or a projection for
the 694-slide cohort. The separate MIL environment is
`.venv\lung-wsi-mil`; the prior augmentation environment was preserved.

## Acceptance results

| Check | Result |
| --- | --- |
| Genuine frozen pilot slides | 14 read and seen; labels 7 Normal / 7 Tumor |
| Patches / stored image views | 873 patches / 2,619 Original+Aug1+Aug2 patch images |
| MIL view-bags | 42 compared to source and 42 full successful forwards |
| Bag sizes | Minimum 18, maximum 217 instances; no patch subsampling |
| DataLoader batches | `[12,147,1024]`, `[12,217,1024]`, `[12,48,1024]`, `[6,45,1024]` |
| Padding | 2,595 padded rows checked; aligned `-Inf`/`[-1,-1]` tail sentinels |
| Source fidelity | Every trimmed view exactly matched its saved embedding after the frozen Dataset dtype conversion |
| Outputs | All inputs/outputs finite; attention nonnegative and normalized |
| AMIL acceptance suite | 34 run: 33 passed, 1 optional MPS skip; zero failures/errors |
| Existing data regressions | 15 passed; zero failures/skips |
| Full quality tests | 163 passed, 1 optional MPS skip |
| Other quality checks | Compile, frozen-source validation, recorded audit replay, dependency consistency and committed/working-tree whitespace checks all PASSED |

The suite counts overlap: the full quality run includes the AMIL and data-layer
tests. All six acceptance fields in the final AMIL JSON are true, including
official-encoder/frozen-cohort provenance and a stable, identified HDF5 file.

The Dataset/collate/DataLoader interfaces and the three-output AMIL API remain
unchanged: `(Y_prob, Y_hat, A)`. AMIL receives one finite unpadded bag;
validation-only padding removal remains in the tests. Unit tests independently
check attention and weighted aggregation arithmetic, threshold equality at 0.5,
permutation behavior, gradients, initialization and invalid inputs. The AMIL
weights used by this acceptance run were initialized for interface checks;
they are not a trained cancer classifier.

## Artifact identities and retained evidence

The server evidence root is:

```text
D:\Meraj Rastegar\Data\AMIL_Validation_20261017
```

Its feature cache is `pilot14-kimianet.h5`, 10,238,196 bytes. The cache remained
unchanged during validation. The checkpoint was obtained directly from
[upstream KimiaNet](https://github.com/KimiaLabMayo/KimiaNet/tree/8a2f120999e04e4e18e0057bacbd1b47c27a56be)
at source commit `8a2f120999e04e4e18e0057bacbd1b47c27a56be`; its stored location
is `D:\Meraj Rastegar\Data\weights\KimiaNetPyTorchWeights.pth`. Weights, HDF5
data and full raw reports remain outside the repository. Upstream weight terms
and the reproducible commands are documented in
[the acceptance procedure](amil-validation.md).

| Identity | SHA256 / fingerprint |
| --- | --- |
| Feature cache SHA256 | `fe68ee8b91e99cb48e2b774550f31f3a9495dafa1035e8169d280827e216c047` |
| Feature provenance fingerprint | `f2dd073a10e4b9d60e5dec7d0848bf632a34f3de65ef5c2036bf76e966913548` |
| Official weights SHA256 | `f4972513e308d3197fb940177ca607bb9bb8f82690eadaec1e76190ef8b68375` |
| Frozen pilot selection SHA256 | `297da8c7d4a902f42727c6dc54635c9f5558600ec3f5baf4a267d335a1b73d65` |
| Frozen inventory SHA256 | `33ef5ffa9710752ee94944511355c4a0bafa573788e2f48c045b978e32463916` |

The following hashes identify the retained report copies; paths are relative to
the server evidence root. They provide references without publishing the full
reports or their duplicated test/source inventories.

| Evidence file | SHA256 |
| --- | --- |
| `amil-validation.json` | `d0a2bad9c782d5461a1c43a5e74e9d764a70aa66ba3e62f0f360094ca792c8a8` |
| `pilot14-extraction-execution.json` | `266c1a61c62f4954e7a76cfe29de30ba4268694f36bc59b4a55d69cc6c1f9eee` |
| `pilot14-extraction.stdout.log` | `daaa6134fc8ef7e099d918c206ce0ba6e97853331a10d0eb19e3f5a6c1b3a1e8` |
| `cohort-reconciliation.json` | `b2741dda4296f4ddf6355865771f49a073f9f9ef63b7ed3cb608afee1c717d0f` |
| `environment-cu130.txt` | `4d6334e0e5e69d3b7a7b2c36eb93b591a06d379e6208f0e568ccfceeffb86aa3` |
| `quality-server/quality_check_summary.json` | `a10bdd5a716048707023c43b27070133e1fe87388c68d5624c22deea56812668` |
| `quality-server.stdout.log` | `07fb6da5e39fb1fce9f7eacd79d25395d09d81e4fd39f5c576be7d55df57c538` |

A [compact machine-readable summary](runs/amil-validation-2026-10-17.json)
records the essential counts and identities. The retained full AMIL report
contains named test outcomes, tested file hashes, all 14 slide records, source
provenance and warnings.

## Next planned work

Benchmark the genuine streaming encoder on the target server and then extract
features for the full reconciled 694-slide cohort. Before training, finalize
the mean-pooling baseline and AMIL comparison protocol around the frozen
patient-level splits, view/augmentation policy, loss and model-selection rules.
Production padding removal belongs to that training-loop task. No optimizer
step, trained checkpoint or diagnostic metric is delivered by this acceptance
run.
