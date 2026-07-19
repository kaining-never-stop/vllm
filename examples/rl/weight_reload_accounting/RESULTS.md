# Validation Results

## Focused Regression Suite

The CPU-focused suite covers the concrete layerwise duplicate case, a legitimate
two-shard packed load, and the Plan-Apply-Certify failure paths.

```text
26 passed, 34 deselected
ruff: passed
compileall: passed
git diff --check: passed
```

The red revision accepts two applications of `left`, reaches the expected eight
elements, and commits an uninitialized `right`. The guarded revision rejects the
second `left` application before finalization. A separate packed-loader test confirms
that `q` and `k` applications to one destination remain distinct.

## Accounting Microbenchmark

The benchmark ran for ten repetitions on Python 3.12.13 on an Apple Silicon host.
Times are milliseconds; cyclic garbage collection is disabled only inside each timed
window. Full output is in [benchmark-results.json](benchmark-results.json).

| Parameters | Manifest p50/p95 | Plan p50/p95 | Rank receipt p50/p95 |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.523 / 0.628 | 1.553 / 1.618 | 1.029 / 1.144 |
| 10,000 | 5.498 / 5.709 | 16.067 / 17.194 | 10.781 / 11.287 |
| 50,000 | 30.735 / 32.242 | 88.148 / 89.390 | 58.497 / 59.236 |

At 50,000 parameters, certification p95 was 0.036 ms for one rank, 0.037 ms for
two ranks, and 0.065 ms for eight ranks. The corresponding serialized certificates
were 323, 392, and 806 bytes. The rank-local receipt was 347 bytes because it carries
a count and digest rather than every application record.

The first implementation rebuilt the plan index for every application and therefore
had O(N^2) receipt construction. The measured implementation builds that index once
per rank, making manifest, plan, and receipt construction linear in parameter count.

## Two-GPU Evidence

A pinned two-H20, TP=2 run exercised real cold-B and warm-A-to-B reload RPCs on both
main (`e94243893d`) and the session implementation from PR #48908 (`71a5d8b62a`).
Each comparison covered 38 log probabilities, matched token IDs, and observed a
maximum absolute delta of 0.0. Reload time was 0.184 seconds on main and 0.154 seconds
on the session revision.

An independent NCCL probe produced distinct rank receipts and one all-rank certificate;
missing-rank, incomplete-receipt, and divergent-manifest cases were rejected. The
sanitized machine-readable result is in [stage-c-summary.json](stage-c-summary.json).

This is boundary evidence, not an end-to-end integration claim. The real reload RPC
did not emit the prototype receipt, and the protocol probe did not call the real model
loader. The run also emitted a RotaryEmbedding load warning, so it does not establish
complete loader coverage across architectures.
