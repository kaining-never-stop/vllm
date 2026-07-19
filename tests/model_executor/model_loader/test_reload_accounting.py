# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest

from vllm.model_executor.model_loader.reload.accounting import (
    CertificationError,
    LoadTarget,
    ManifestError,
    PlanError,
    ReceiptBuilder,
    ReceiptError,
    SourceParam,
    WeightLoadPlan,
    WeightUpdateManifest,
    certify_weight_update,
)


def make_manifest(transaction_id: str = "update-7") -> WeightUpdateManifest:
    params = (
        SourceParam("q_proj", "float16", (8, 8)),
        SourceParam("k_proj", "float16", (4, 8)),
        SourceParam("v_proj", "float16", (4, 8)),
    )
    return WeightUpdateManifest(
        transaction_id,
        params,
        tuple(param.name for param in params),
    )


def make_plan(manifest: WeightUpdateManifest) -> WeightLoadPlan:
    return WeightLoadPlan.compile(
        manifest,
        lambda source: LoadTarget("qkv_proj", source.name[0]),
    )


def close_receipt(plan: WeightLoadPlan, rank: int = 0):
    builder = ReceiptBuilder(plan, rank)
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


def test_manifest_requires_exact_terminal_classification():
    with pytest.raises(ManifestError, match="classification mismatch"):
        WeightUpdateManifest(
            "partial",
            (SourceParam("q_proj", "float16", (8, 8)),),
            ("q_proj", "v_proj"),
        )


def test_manifest_accepts_explicit_preserve_and_allowed_missing():
    manifest = WeightUpdateManifest(
        "partial",
        (SourceParam("q_proj", "float16", (8, 8)),),
        ("q_proj", "k_proj", "v_proj"),
        preserved_keys=("k_proj",),
        allowed_missing_keys=("v_proj",),
    )
    assert manifest.required_keys == ("k_proj", "q_proj", "v_proj")


def test_manifest_rejects_duplicate_source_names():
    duplicate = SourceParam("q_proj", "float16", (8, 8))
    with pytest.raises(ManifestError, match="duplicate"):
        WeightUpdateManifest("duplicate", (duplicate, duplicate), ("q_proj",))


def test_plan_requires_explicit_shards_for_shared_destination():
    with pytest.raises(PlanError, match="explicit shard"):
        WeightLoadPlan.compile(make_manifest(), lambda _: LoadTarget("qkv_proj"))


def test_plan_rejects_duplicate_destination_shard_mapping():
    with pytest.raises(PlanError, match="duplicate destination"):
        WeightLoadPlan.compile(
            make_manifest(), lambda _: LoadTarget("qkv_proj", "same")
        )


def test_complete_rank_receipt_closes():
    plan = make_plan(make_manifest())
    receipt = close_receipt(plan)
    assert receipt.application_count == 3
    assert len(receipt.application_digest) == 64


def test_duplicate_application_is_rejected_immediately():
    plan = make_plan(make_manifest())
    builder = ReceiptBuilder(plan, 0)
    entry = plan.entries[0]
    kwargs = {
        "shard_id": entry.shard_id,
        "source_dtype": entry.source.dtype,
        "source_shape": entry.source.shape,
    }
    builder.record(entry.source.name, entry.destination_name, **kwargs)
    with pytest.raises(ReceiptError, match="duplicate source"):
        builder.record(entry.source.name, entry.destination_name, **kwargs)


def test_missing_application_is_rejected_on_close():
    plan = make_plan(make_manifest())
    builder = ReceiptBuilder(plan, 0)
    entry = plan.entries[0]
    builder.record(
        entry.source.name,
        entry.destination_name,
        shard_id=entry.shard_id,
        source_dtype=entry.source.dtype,
        source_shape=entry.source.shape,
    )
    with pytest.raises(ReceiptError, match="missing source"):
        builder.finish()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("destination_name", "wrong"),
        ("shard_id", "wrong"),
        ("source_dtype", "float32"),
        ("source_shape", (64,)),
        ("applied_dtype", "float32"),
        ("applied_shape", (64,)),
    ],
)
def test_application_must_match_immutable_plan(field, value):
    plan = make_plan(make_manifest())
    entry = plan.entries[0]
    values = {
        "destination_name": entry.destination_name,
        "shard_id": entry.shard_id,
        "source_dtype": entry.source.dtype,
        "source_shape": entry.source.shape,
        "applied_dtype": entry.applied_dtype,
        "applied_shape": entry.applied_shape,
    }
    values[field] = value
    destination_name = values.pop("destination_name")
    with pytest.raises(ReceiptError, match="does not match plan"):
        ReceiptBuilder(plan, 0).record(entry.source.name, destination_name, **values)


def test_record_after_close_is_rejected():
    plan = make_plan(make_manifest())
    builder = ReceiptBuilder(plan, 0)
    for entry in plan.entries:
        builder.record(
            entry.source.name,
            entry.destination_name,
            shard_id=entry.shard_id,
            source_dtype=entry.source.dtype,
            source_shape=entry.source.shape,
        )
    builder.finish()
    with pytest.raises(ReceiptError, match="after receipt closure"):
        entry = plan.entries[0]
        builder.record(
            entry.source.name,
            entry.destination_name,
            shard_id=entry.shard_id,
            source_dtype=entry.source.dtype,
            source_shape=entry.source.shape,
        )


def test_two_rank_receipts_create_order_independent_certificate():
    manifest = make_manifest()
    plan = make_plan(manifest)
    rank0 = close_receipt(plan, 0)
    rank1 = close_receipt(plan, 1)
    first = certify_weight_update(manifest, plan, (rank0, rank1), expected_ranks=(0, 1))
    second = certify_weight_update(
        manifest, plan, (rank1, rank0), expected_ranks=(1, 0)
    )
    assert first.digest == second.digest
    assert rank0.digest != rank1.digest


def test_certificate_rejects_missing_rank():
    manifest = make_manifest()
    plan = make_plan(manifest)
    with pytest.raises(CertificationError, match=r"missing=\[1\]"):
        certify_weight_update(
            manifest, plan, (close_receipt(plan, 0),), expected_ranks=(0, 1)
        )


def test_certificate_rejects_duplicate_rank():
    manifest = make_manifest()
    plan = make_plan(manifest)
    receipt = close_receipt(plan, 0)
    with pytest.raises(CertificationError, match="duplicate rank"):
        certify_weight_update(manifest, plan, (receipt, receipt), expected_ranks=(0,))


def test_certificate_rejects_divergent_manifest_or_plan():
    manifest = make_manifest()
    plan = make_plan(manifest)
    receipt = close_receipt(plan, 0)
    with pytest.raises(CertificationError, match="manifest mismatch"):
        certify_weight_update(
            manifest,
            plan,
            (replace(receipt, manifest_digest="0" * 64),),
            expected_ranks=(0,),
        )
    with pytest.raises(CertificationError, match="plan mismatch"):
        certify_weight_update(
            manifest,
            plan,
            (replace(receipt, plan_digest="f" * 64),),
            expected_ranks=(0,),
        )


@pytest.mark.parametrize("expected_ranks", [(), (0, 0), (-1,), ("0",)])
def test_certificate_rejects_invalid_expected_ranks(expected_ranks):
    manifest = make_manifest()
    plan = make_plan(manifest)
    with pytest.raises(CertificationError, match="unique and non-negative"):
        certify_weight_update(
            manifest,
            plan,
            (close_receipt(plan, 0),),
            expected_ranks=expected_ranks,
        )


def test_certificate_rejects_invalid_receipt_summary():
    manifest = make_manifest()
    plan = make_plan(manifest)
    receipt = close_receipt(plan, 0)
    with pytest.raises(CertificationError, match="application count"):
        certify_weight_update(
            manifest,
            plan,
            (replace(receipt, application_count=receipt.application_count - 1),),
            expected_ranks=(0,),
        )
    with pytest.raises(CertificationError, match="application digest"):
        certify_weight_update(
            manifest,
            plan,
            (replace(receipt, application_digest="z" * 64),),
            expected_ranks=(0,),
        )
