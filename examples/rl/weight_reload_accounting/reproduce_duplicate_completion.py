# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the element-count completion gap in layerwise weight reload."""

import argparse
import json

import torch

from vllm.model_executor.model_loader.reload import meta as reload_meta
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)


class DuplicateCompletionLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.zeros(4))
        self.right = torch.nn.Parameter(torch.zeros(4))


def materialize_with_sentinel(meta_tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        requires_grad=False,
    )
    tensor.fill_(float("nan"))
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__.copy()
    return tensor


def reproduce() -> dict[str, object]:
    layer = DuplicateCompletionLayer()
    model = torch.nn.Sequential(layer)
    original_materializer = reload_meta.materialize_meta_tensor
    reload_meta.materialize_meta_tensor = materialize_with_sentinel
    error: str | None = None
    try:
        record_metadata_for_reloading(model)
        initialize_layerwise_reload(model)
        layer.left.weight_loader(layer.left, torch.ones(4))
        layer.left.weight_loader(layer.left, torch.full((4,), 2.0))
        finalize_layerwise_reload(model, model_config=None)
    except ValueError as exc:
        error = str(exc)
    finally:
        reload_meta.materialize_meta_tensor = original_materializer

    left_values = None if layer.left.is_meta else layer.left.detach().tolist()
    right_is_all_nan = (
        None if layer.right.is_meta else bool(torch.isnan(layer.right).all())
    )
    return {
        "duplicate_rejected": error is not None,
        "error": error,
        "left": left_values,
        "right_is_all_nan": right_is_all_nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("vulnerable", "guarded"), required=True)
    args = parser.parse_args()
    result = reproduce()
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.expect == "vulnerable":
        matched = not result["duplicate_rejected"] and result["right_is_all_nan"]
    else:
        matched = result["duplicate_rejected"] and "duplicate" in str(result["error"])
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
