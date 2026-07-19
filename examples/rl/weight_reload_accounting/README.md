# Weight Reload Completion Accounting Prototype

This directory isolates a completion-accounting gap found while validating repeated
weight reloads for online RL. The current layerwise path uses copied element counts
to decide that a layer is complete. Two equal-size applications of one parameter can
therefore reach the expected count while a sibling parameter is absent.

## Concrete Reproduction

The `issue-48312-red` tag contains a regression test on the unguarded implementation.
The same test passes on this branch after applications are identified by parameter and
loader shard selectors.

```bash
git switch --detach issue-48312-red
python -m pytest -q --confcutdir=tests/model_executor/model_loader \
  tests/model_executor/model_loader/test_reload.py \
  -k duplicate_that_masks_missing_sibling

git switch issue-48312-weight-reload-commit-certificate
python -m pytest -q --confcutdir=tests/model_executor/model_loader \
  tests/model_executor/model_loader/test_reload.py \
  -k 'duplicate_that_masks_missing_sibling or distinct_packed_shard_applications'
```

For a standalone state report, keep the script outside the checkout while switching
between the two revisions:

```bash
cp examples/rl/weight_reload_accounting/reproduce_duplicate_completion.py /tmp/
git switch --detach issue-48312-red
PYTHONPATH=. python /tmp/reproduce_duplicate_completion.py --expect vulnerable
git switch issue-48312-weight-reload-commit-certificate
PYTHONPATH=. python /tmp/reproduce_duplicate_completion.py --expect guarded
```

## Proposed Contract

The broader prototype separates completion into four explicit records:

1. `WeightUpdateManifest` classifies every expected source key as transmitted,
   preserved, or explicitly allowed to be missing.
2. `WeightLoadPlan` resolves each transmitted key to one destination and optional
   packed/expert shard before any payload is applied.
3. `ReceiptBuilder` rejects unexpected, duplicate, missing, shape-divergent, and
   dtype-divergent rank-local applications.
4. `certify_weight_update` accepts only the exact expected rank set with one receipt
   per rank and consistent transaction, manifest, and plan digests.

The receipt and certificate contain only metadata digests and counts. They do not hash
tensor contents, replace transport checks, or protect against a malicious process.
The accounting module is intentionally not wired into `WeightLoadSession` yet; that
return boundary should be agreed with maintainers before changing the public worker
and control-plane APIs.

## Validation

```bash
python -m pytest -q --confcutdir=tests/model_executor/model_loader \
  tests/model_executor/model_loader/test_reload_accounting.py \
  tests/model_executor/model_loader/test_reload.py \
  -k 'reload_accounting or duplicate_that_masks_missing_sibling or distinct_packed_shard_applications'

PYTHONPATH=. python examples/rl/weight_reload_accounting/benchmark.py --repetitions 10
```

See [RESULTS.md](RESULTS.md) for the focused CPU checks, protocol microbenchmark, and
sanitized two-GPU reload evidence. The implementation and write-up were prepared with
assistance from OpenAI Codex and were reviewed and validated by the author.
