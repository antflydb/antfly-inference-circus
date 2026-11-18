"""Observed checkpoint/framework facts and executable-bound native source evidence."""
from __future__ import annotations

import collections
import functools
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any


def _count_parameters(parameters) -> dict:
    dtypes, devices, components = collections.Counter(), collections.Counter(), {}
    for name, parameter in parameters:
        elements = int(parameter.numel())
        dtype, device = str(parameter.dtype), str(parameter.device)
        dtypes[dtype] += elements
        devices[device] += elements
        component = "audio" if "audio" in name else "vision" if "vision" in name else "text_and_shared"
        entry = components.setdefault(component, {"elements": 0, "bytes": 0})
        entry["elements"] += elements
        entry["bytes"] += elements * int(parameter.element_size())
    return {"parameter_elements_by_dtype": dict(dtypes), "parameter_elements_by_device": dict(devices),
            "loaded_parameter_components": components}


def reference_precision(model, *, requested_dtype=None, default_policy="library_default", override_reason=None) -> dict:
    import torch

    result = {"evidence_kind": "runtime_framework_observation", "requested_dtype": requested_dtype,
              "framework": {"name": "PyTorch", "version": str(torch.__version__),
                            "reported_source_commit": torch.version.git_version},
              "loading_policy": default_policy, "override_reason": override_reason,
              **_count_parameters(model.named_parameters()),
              "observation_scope": "outside forward at model load; internal autocast and kernel arithmetic not traced",
              "parameter_storage_is_not_compute_dtype": True,
              "float32_matmul_precision": torch.get_float32_matmul_precision()}
    autocast = {}
    for device in ("cpu", "cuda", "mps"):
        try:
            autocast[device] = {"enabled": torch.is_autocast_enabled(device),
                                "dtype": str(torch.get_autocast_dtype(device))}
        except (RuntimeError, TypeError, AttributeError) as error:
            autocast[device] = {"available": False, "reason": type(error).__name__}
    result["autocast"] = autocast
    result["cuda_math_policy"] = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    return result


def safetensors_inventory(path: Path) -> dict:
    """Inspect actual tensor headers without touching all model pages during measurement."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 32 << 20:
            raise ValueError("invalid safetensors header length")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("truncated safetensors header")
        header = json.loads(payload)
    counts, elements = collections.Counter(), collections.Counter()
    data_length = path.stat().st_size - 8 - length
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape, offsets, dtype = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= data_length):
            raise ValueError("invalid safetensors descriptor")
        counts[dtype] += 1
        elements[dtype] += math.prod(shape)
    return {"name": path.name, "size_bytes": path.stat().st_size,
            "tensor_count_by_dtype": dict(counts), "elements_by_dtype": dict(elements),
            "evidence_kind": "observed_safetensors_header"}


@functools.lru_cache(maxsize=32)
def _binary_digest(path: str, size: int, mtime_ns: int) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def executable_digest(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    return _binary_digest(str(path), stat.st_size, stat.st_mtime_ns)


def checkpoint_files(root: Path, model: str) -> list[Path]:
    """Resolve an installed catalog model by its manifest, not an ambiguous prefix."""
    owner, name = model.split("/", 1)
    candidates = []
    parent = root / owner
    if parent.is_dir():
        for directory in parent.iterdir():
            if not directory.is_dir() or not (directory.name == name or directory.name.startswith(name + "--")):
                continue
            receipt = directory / ".antfly-download-complete.json"
            if receipt.is_file():
                record = json.loads(receipt.read_text())
                source = record.get("source", {})
                if source.get("owner") != owner or source.get("name") != name:
                    continue
            candidates.append(directory)
    if len(candidates) > 1:
        raise ValueError(f"ambiguous installed model: {model}")
    if not candidates:
        return []
    return sorted(candidates[0].glob("*.safetensors"))


def load_native_evidence(path: Path, binary_sha256: str, model: str, backend: str) -> dict | None:
    if not path.is_file():
        return None
    encoded = path.read_bytes()
    record = json.loads(encoded)
    if record.get("schema_version") != 1:
        raise ValueError("unsupported native precision evidence schema")
    bindings = record.get("executables", {})
    binding = bindings.get(binary_sha256)
    if binding is None:
        return None
    commit = binding.get("source_commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("native precision evidence lacks an exact source commit")
    policy = binding.get("models", {}).get(model, {}).get(backend)
    if policy is None:
        return None
    sources = policy.get("sources", [])
    if not sources or any(f"/{commit}/" not in source for source in sources):
        raise ValueError("native precision sources are not pinned to the bound executable commit")
    return {**policy, "evidence_kind": "executable_bound_source_policy", "source_commit": commit,
            "binary_sha256": binary_sha256, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def native_precision(binary: Path, model: str, models_dir: Path, backend: str) -> dict[str, Any]:
    digest = executable_digest(binary)
    files = checkpoint_files(models_dir, model)
    inventory = [safetensors_inventory(path) for path in files]
    evidence_path = Path(os.environ.get("TERMITE_NATIVE_EVIDENCE", Path(__file__).with_name("native_precision_contracts.json")))
    source = load_native_evidence(evidence_path, digest, model, backend)
    return {
        "binary_sha256": digest, "model": model, "backend": backend,
        "checkpoint_storage": inventory,
        "native_policy": source or {"evidence_kind": "unavailable", "reason": "no source policy bound to this executable/model/backend"},
        "runtime_kernel_trace_observed": False,
        "environment_overrides": {key: value for key, value in sorted(os.environ.items())
                                  if key.startswith(("ANTFLY_CUDA_", "TERMITE_METAL_", "ANTFLY_METAL_"))
                                  and key != "TERMITE_METAL_MAPPING_EVIDENCE"},
    }
