# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark the Plan-Apply-Certify accounting prototype."""

import argparse
import gc
import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from vllm.model_executor.model_loader.reload.accounting import (
    LoadTarget,
    ReceiptBuilder,
    SourceParam,
    WeightLoadPlan,
    WeightUpdateManifest,
    certify_weight_update,
)


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def measure(call: Callable[[], Any], repetitions: int) -> dict[str, float]:
    call()
    samples: list[float] = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "p50_ms": round(percentile(samples, 0.50), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "min_ms": round(min(samples), 3),
    }


def canonical_size(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def build_receipt(plan: WeightLoadPlan):
    builder = ReceiptBuilder(plan, rank=0)
    for entry in plan.entries:
        builder.record(
            entry.source.name,
            entry.destination_name,
            shard_id=entry.shard_id,
            source_dtype=entry.source.dtype,
            source_shape=entry.source.shape,
            applied_dtype=entry.applied_dtype,
            applied_shape=entry.applied_shape,
        )
    return builder.finish()


def benchmark_case(parameter_count: int, repetitions: int) -> dict[str, object]:
    params = tuple(
        SourceParam(f"layers.{index}.weight", "bfloat16", (128, 128))
        for index in range(parameter_count)
    )
    required = tuple(param.name for param in params)

    def make_manifest() -> WeightUpdateManifest:
        return WeightUpdateManifest("benchmark-update", params, required)

    manifest_latency = measure(make_manifest, repetitions)
    manifest = make_manifest()

    def make_plan() -> WeightLoadPlan:
        return WeightLoadPlan.compile(manifest, lambda source: LoadTarget(source.name))

    plan_latency = measure(make_plan, repetitions)
    plan = make_plan()
    receipt_latency = measure(lambda: build_receipt(plan), repetitions)
    receipt = build_receipt(plan)

    certificates: dict[str, object] = {}
    for rank_count in (1, 2, 8):
        receipts = tuple(replace(receipt, rank=rank) for rank in range(rank_count))

        def certify(receipts=receipts, rank_count=rank_count):
            return certify_weight_update(
                manifest,
                plan,
                receipts,
                expected_ranks=range(rank_count),
            )

        latency = measure(certify, repetitions)
        certificate = certify()
        certificates[str(rank_count)] = {
            **latency,
            "serialized_bytes": canonical_size(certificate.to_dict()),
        }

    return {
        "parameter_count": parameter_count,
        "manifest": manifest_latency,
        "plan": plan_latency,
        "rank_local_receipt": {
            **receipt_latency,
            "serialized_bytes": canonical_size(receipt.to_dict()),
        },
        "certificate_by_rank_count": certificates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--sizes", default="1000,10000,50000")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if any(size < 1 for size in sizes):
        parser.error("--sizes must contain positive integers")

    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repetitions": args.repetitions,
        "cases": [benchmark_case(size, args.repetitions) for size in sizes],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
