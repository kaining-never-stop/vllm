# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed accounting primitives for distributed weight updates.

The protocol separates the expected source set, its loader-specific mapping,
rank-local application evidence, and the distributed commit decision. It does
not hash tensor contents or prescribe a transport.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import ParamMeta

PROTOCOL_VERSION = 1


class WeightUpdateAccountingError(ValueError):
    """Base class for malformed or incomplete update evidence."""


class ManifestError(WeightUpdateAccountingError):
    """The expected source set is ambiguous."""


class PlanError(WeightUpdateAccountingError):
    """The loader cannot resolve the expected source set exactly once."""


class ReceiptError(WeightUpdateAccountingError):
    """A rank-local application differs from the immutable plan."""


class CertificationError(WeightUpdateAccountingError):
    """The distributed receipt set cannot be committed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _normalize_names(values: Iterable[str], label: str) -> tuple[str, ...]:
    names = tuple(values)
    if any(not name for name in names):
        raise ManifestError(f"{label} must contain non-empty names")
    if len(set(names)) != len(names):
        raise ManifestError(f"{label} contains duplicate names")
    return tuple(sorted(names))


def _dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


@dataclass(frozen=True, order=True)
class SourceParam:
    """Canonical snapshot of one trainer-side ``ParamMeta`` value."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_param_meta(cls, value: ParamMeta) -> SourceParam:
        return cls(value.name, _dtype_name(value.dtype), tuple(value.shape))

    def __post_init__(self) -> None:
        if not self.name or not self.dtype:
            raise ManifestError("source name and dtype must be non-empty")
        shape = tuple(self.shape)
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ManifestError(f"invalid source shape for {self.name!r}")
        object.__setattr__(self, "shape", shape)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class WeightUpdateManifest:
    """Immutable classification of every key in an update transaction."""

    transaction_id: str
    params: tuple[SourceParam, ...]
    required_keys: tuple[str, ...]
    preserved_keys: tuple[str, ...] = ()
    allowed_missing_keys: tuple[str, ...] = ()
    _digest: str = field(init=False, repr=False, compare=False)

    @classmethod
    def from_param_meta(
        cls,
        transaction_id: str,
        params: Sequence[ParamMeta],
        *,
        required_keys: Iterable[str] | None = None,
        preserved_keys: Iterable[str] = (),
        allowed_missing_keys: Iterable[str] = (),
    ) -> WeightUpdateManifest:
        snapshots = tuple(SourceParam.from_param_meta(param) for param in params)
        required = (
            tuple(param.name for param in snapshots)
            if required_keys is None
            else tuple(required_keys)
        )
        return cls(
            transaction_id,
            snapshots,
            required,
            tuple(preserved_keys),
            tuple(allowed_missing_keys),
        )

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ManifestError("transaction_id must be non-empty")
        params = tuple(sorted(self.params, key=lambda item: item.name))
        source_names = _normalize_names(
            (param.name for param in params), "source params"
        )
        if not source_names:
            raise ManifestError("manifest must contain at least one source param")
        required = _normalize_names(self.required_keys, "required_keys")
        preserved = _normalize_names(self.preserved_keys, "preserved_keys")
        missing = _normalize_names(self.allowed_missing_keys, "allowed_missing_keys")
        classifications = (set(source_names), set(preserved), set(missing))
        if any(
            left & right
            for index, left in enumerate(classifications)
            for right in classifications[index + 1 :]
        ):
            raise ManifestError("key classifications must be disjoint")
        classified = set().union(*classifications)
        if classified != set(required):
            absent = sorted(set(required) - classified)
            extra = sorted(classified - set(required))
            raise ManifestError(
                f"required key classification mismatch: missing={absent}, extra={extra}"
            )
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "required_keys", required)
        object.__setattr__(self, "preserved_keys", preserved)
        object.__setattr__(self, "allowed_missing_keys", missing)
        object.__setattr__(self, "_digest", _digest(self.to_dict()))

    @property
    def digest(self) -> str:
        return self._digest

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transaction_id": self.transaction_id,
            "params": [param.to_dict() for param in self.params],
            "required_keys": list(self.required_keys),
            "preserved_keys": list(self.preserved_keys),
            "allowed_missing_keys": list(self.allowed_missing_keys),
        }


@dataclass(frozen=True)
class LoadTarget:
    destination_name: str
    shard_id: str | int | None = None
    applied_dtype: str | None = None
    applied_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.destination_name:
            raise PlanError("destination_name must be non-empty")
        if self.applied_shape is not None:
            shape = tuple(self.applied_shape)
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise PlanError(f"invalid applied shape for {self.destination_name!r}")
            object.__setattr__(self, "applied_shape", shape)


@dataclass(frozen=True)
class PlanEntry:
    source: SourceParam
    destination_name: str
    shard_id: str | int | None
    applied_dtype: str
    applied_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "destination_name": self.destination_name,
            "shard_id": self.shard_id,
            "applied_dtype": self.applied_dtype,
            "applied_shape": list(self.applied_shape),
        }


Resolver = Callable[[SourceParam], LoadTarget | None]


@dataclass(frozen=True)
class WeightLoadPlan:
    transaction_id: str
    manifest_digest: str
    entries: tuple[PlanEntry, ...]
    _digest: str = field(init=False, repr=False, compare=False)

    @classmethod
    def compile(
        cls, manifest: WeightUpdateManifest, resolver: Resolver
    ) -> WeightLoadPlan:
        entries: list[PlanEntry] = []
        target_keys: set[tuple[str, str | int | None]] = set()
        destination_shards: dict[str, set[str | int | None]] = {}
        for source in manifest.params:
            target = resolver(source)
            if target is None:
                raise PlanError(f"unresolved source key: {source.name}")
            key = (target.destination_name, target.shard_id)
            previous_shards = destination_shards.get(target.destination_name, set())
            if previous_shards and (target.shard_id is None or None in previous_shards):
                raise PlanError(
                    "shared destinations require explicit shard ids: "
                    f"{target.destination_name}"
                )
            if key in target_keys:
                raise PlanError(
                    "duplicate destination/shard mapping: "
                    f"{target.destination_name}[{target.shard_id!r}]"
                )
            target_keys.add(key)
            destination_shards.setdefault(target.destination_name, set()).add(
                target.shard_id
            )
            entries.append(
                PlanEntry(
                    source,
                    target.destination_name,
                    target.shard_id,
                    target.applied_dtype or source.dtype,
                    target.applied_shape or source.shape,
                )
            )
        return cls(
            manifest.transaction_id,
            manifest.digest,
            tuple(sorted(entries, key=lambda item: item.source.name)),
        )

    def __post_init__(self) -> None:
        if not self.entries:
            raise PlanError("plan must contain at least one load entry")
        object.__setattr__(self, "_digest", _digest(self.to_dict()))

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def by_source(self) -> dict[str, PlanEntry]:
        return {entry.source.name: entry for entry in self.entries}

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transaction_id": self.transaction_id,
            "manifest_digest": self.manifest_digest,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class WeightApplyReceipt:
    transaction_id: str
    rank: int
    manifest_digest: str
    plan_digest: str
    application_count: int
    application_digest: str

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transaction_id": self.transaction_id,
            "rank": self.rank,
            "manifest_digest": self.manifest_digest,
            "plan_digest": self.plan_digest,
            "application_count": self.application_count,
            "application_digest": self.application_digest,
        }


class ReceiptBuilder:
    """Record one rank's applications and close them against a fixed plan."""

    def __init__(self, plan: WeightLoadPlan, rank: int) -> None:
        if not isinstance(rank, int) or rank < 0:
            raise ReceiptError("rank must be non-negative")
        self._plan = plan
        self._by_source = plan.by_source
        self._rank = rank
        self._applications: list[dict[str, object]] = []
        self._seen: set[str] = set()
        self._closed = False

    def record(
        self,
        source_name: str,
        destination_name: str,
        *,
        shard_id: str | int | None = None,
        source_dtype: str,
        source_shape: Iterable[int],
        applied_dtype: str | None = None,
        applied_shape: Iterable[int] | None = None,
    ) -> None:
        if self._closed:
            raise ReceiptError("cannot record after receipt closure")
        planned = self._by_source.get(source_name)
        if planned is None:
            raise ReceiptError(f"unexpected source application: {source_name}")
        if source_name in self._seen:
            raise ReceiptError(f"duplicate source application: {source_name}")
        source_shape_tuple = tuple(source_shape)
        applied_shape_tuple = (
            source_shape_tuple if applied_shape is None else tuple(applied_shape)
        )
        actual = (
            destination_name,
            shard_id,
            source_dtype,
            source_shape_tuple,
            applied_dtype or source_dtype,
            applied_shape_tuple,
        )
        expected = (
            planned.destination_name,
            planned.shard_id,
            planned.source.dtype,
            planned.source.shape,
            planned.applied_dtype,
            planned.applied_shape,
        )
        if actual != expected:
            raise ReceiptError(
                f"application does not match plan for {source_name}: "
                f"expected={expected!r}, actual={actual!r}"
            )
        self._seen.add(source_name)
        self._applications.append(
            {
                "source_name": source_name,
                "destination_name": destination_name,
                "shard_id": shard_id,
                "source_dtype": source_dtype,
                "source_shape": list(source_shape_tuple),
                "applied_dtype": applied_dtype or source_dtype,
                "applied_shape": list(applied_shape_tuple),
            }
        )

    def finish(self) -> WeightApplyReceipt:
        if self._closed:
            raise ReceiptError("receipt is already closed")
        self._closed = True
        missing = sorted(set(self._by_source) - self._seen)
        if missing:
            raise ReceiptError(f"missing source applications: {missing}")
        applications = sorted(
            self._applications, key=lambda item: str(item["source_name"])
        )
        return WeightApplyReceipt(
            self._plan.transaction_id,
            self._rank,
            self._plan.manifest_digest,
            self._plan.digest,
            len(applications),
            _digest(applications),
        )


@dataclass(frozen=True)
class WeightCommitCertificate:
    transaction_id: str
    manifest_digest: str
    plan_digest: str
    ranks: tuple[int, ...]
    receipt_digests: tuple[str, ...]

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transaction_id": self.transaction_id,
            "manifest_digest": self.manifest_digest,
            "plan_digest": self.plan_digest,
            "ranks": list(self.ranks),
            "receipt_digests": list(self.receipt_digests),
        }


def certify_weight_update(
    manifest: WeightUpdateManifest,
    plan: WeightLoadPlan,
    receipts: Iterable[WeightApplyReceipt],
    *,
    expected_ranks: Iterable[int],
) -> WeightCommitCertificate:
    """Commit only a complete, internally consistent set of rank receipts."""
    raw_expected = tuple(expected_ranks)
    if (
        not raw_expected
        or any(not isinstance(rank, int) or rank < 0 for rank in raw_expected)
        or len(set(raw_expected)) != len(raw_expected)
    ):
        raise CertificationError("expected_ranks must be unique and non-negative")
    expected = tuple(sorted(raw_expected))
    if plan.transaction_id != manifest.transaction_id:
        raise CertificationError("plan transaction does not match manifest")
    if plan.manifest_digest != manifest.digest:
        raise CertificationError("plan manifest digest does not match manifest")
    by_rank: dict[int, WeightApplyReceipt] = {}
    for receipt in receipts:
        if receipt.rank in by_rank:
            raise CertificationError(f"duplicate rank receipt: {receipt.rank}")
        by_rank[receipt.rank] = receipt
    observed = tuple(sorted(by_rank))
    if observed != expected:
        raise CertificationError(
            "rank set mismatch: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    for rank in expected:
        receipt = by_rank[rank]
        if receipt.transaction_id != manifest.transaction_id:
            raise CertificationError(f"rank {rank} transaction mismatch")
        if receipt.manifest_digest != manifest.digest:
            raise CertificationError(f"rank {rank} manifest mismatch")
        if receipt.plan_digest != plan.digest:
            raise CertificationError(f"rank {rank} plan mismatch")
        if receipt.application_count != len(plan.entries):
            raise CertificationError(f"rank {rank} application count mismatch")
        if not _is_sha256(receipt.application_digest):
            raise CertificationError(f"rank {rank} application digest is invalid")
    return WeightCommitCertificate(
        manifest.transaction_id,
        manifest.digest,
        plan.digest,
        expected,
        tuple(by_rank[rank].digest for rank in expected),
    )
