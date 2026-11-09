"""Independent CPU block quantization codec, with its complete wire accounting.

This is an integration of scalar quantization, greedy bit allocation and error
feedback bookkeeping.  It makes no novelty, global-optimality or contraction
claim.  The payload is not authenticated and implements no network or chain.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any

import numpy as np
import torch


BLOCK_SIZE = 256
HEADER_SIZE = 64
MAGIC = b"MPQCODEC"
VERSION = 1
HEADER = struct.Struct("<8sIIII40s")
MODES = ("uniform3", "mixed_mse", "mixed_sensitive")


@dataclass(frozen=True)
class EncodedUpdate:
    sent: torch.Tensor
    residual: torch.Tensor
    wire: bytes
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Candidate:
    bitwidth: int
    scale: float
    codes: np.ndarray
    mse: float
    value_bytes: int


def wire_size(numel: int, bitwidth: int) -> int:
    """Exact bytes for a uniform bitwidth, including header and short block."""
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise ValueError("numel must be a positive integer")
    if bitwidth not in (2, 3, 4):
        raise ValueError("bitwidth must be 2, 3, or 4")
    whole, tail = divmod(numel, BLOCK_SIZE)
    nblocks = whole + int(tail != 0)
    value_bytes = whole * (BLOCK_SIZE * bitwidth // 8)
    value_bytes += (tail * bitwidth + 7) // 8
    return HEADER_SIZE + 5 * nblocks + value_bytes


def _input_array(value: torch.Tensor, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must contain at least one coordinate")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    # Explicit FP32 update semantics, independent of the caller's device/dtype.
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError(f"{name} is not representable as finite float32")
    return cpu.numpy().reshape(-1).copy(), tuple(value.shape)


def _importance_array(importance: torch.Tensor | None, numel: int) -> np.ndarray | None:
    if importance is None:
        return None
    if not isinstance(importance, torch.Tensor) or not importance.is_floating_point():
        raise ValueError("importance must be a real floating-point torch.Tensor")
    if importance.numel() != numel:
        raise ValueError("importance must have one value per update coordinate")
    values = importance.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy().reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("importance must be finite and nonnegative")
    return values


def _assign(values64: np.ndarray, scale: np.float32, qmax: int) -> tuple[np.ndarray, np.ndarray]:
    if scale == 0:
        d = np.zeros(values64.size, dtype=np.int16)
    else:
        d = np.clip(np.rint(values64 / float(scale)), -qmax, qmax).astype(np.int16)
    # This FP32 multiplication is the actual decoder operation.
    with np.errstate(over="ignore", invalid="ignore"):
        decoded = d.astype(np.float32) * scale
    return d, decoded


def _quantize(values: np.ndarray, bitwidth: int) -> _Candidate:
    values64 = values.astype(np.float64)
    qmax = (1 << (bitwidth - 1)) - 1
    maxabs = float(np.max(np.abs(values64)))
    if maxabs == 0:
        return _Candidate(bitwidth, 0.0, np.full(values.size, qmax, dtype=np.uint8), 0.0,
                          (values.size * bitwidth + 7) // 8)

    # Evaluate initialization plus exactly four LS/reassignment iterations.
    # Every candidate first goes through its on-wire FP32 scale roundtrip.
    scale = np.float32(maxabs / qmax)
    best: tuple[float, float, np.ndarray] | None = None
    for iteration in range(5):
        d, decoded = _assign(values64, scale, qmax)
        if np.isfinite(decoded).all():
            error = values64 - decoded.astype(np.float64)
            mse = float(np.mean(error * error))
            if best is None or mse < best[0]:
                best = (mse, float(scale), (d + qmax).astype(np.uint8))
        if iteration != 4:
            d64 = d.astype(np.float64)
            denominator = float(np.dot(d64, d64))
            scale = np.float32(float(np.dot(values64, d64)) / denominator) if denominator else np.float32(0)
    if best is None:
        raise ValueError("no finite FP32 decoded quantization candidate")
    return _Candidate(bitwidth, best[1], best[2], best[0], (values.size * bitwidth + 7) // 8)


def _pack_codes(codes: np.ndarray, bitwidth: int) -> bytes:
    bits = ((codes[:, None].astype(np.uint16) >> np.arange(bitwidth)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def decode(wire: bytes) -> torch.Tensor:
    """Strictly decode one canonical payload to a flat CPU FP32 tensor.

    All fields are checked before use; no shape, scale, length or code metadata
    comes from the encoder's in-memory objects.  Trailing bytes are rejected.
    """
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise ValueError("wire must be bytes-like")
    wire = bytes(wire)
    if len(wire) < HEADER_SIZE:
        raise ValueError("truncated header")
    magic, version, numel, block_size, nblocks, reserved = HEADER.unpack_from(wire)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported magic or version")
    if numel == 0 or block_size != BLOCK_SIZE or nblocks != (numel + BLOCK_SIZE - 1) // BLOCK_SIZE:
        raise ValueError("invalid coordinate/block metadata")
    if any(reserved):
        raise ValueError("reserved header bytes must be zero")
    # Reject impossible advertised sizes before allocating the output vector.
    if len(wire) < wire_size(numel, 2) or len(wire) > wire_size(numel, 4):
        raise ValueError("invalid payload length")
    output = np.empty(numel, dtype=np.float32)
    offset = HEADER_SIZE
    for block_index in range(nblocks):
        count = min(BLOCK_SIZE, numel - block_index * BLOCK_SIZE)
        if offset + 5 > len(wire):
            raise ValueError("truncated block metadata")
        bitwidth = wire[offset]
        scale = struct.unpack_from("<f", wire, offset + 1)[0]
        offset += 5
        if bitwidth not in (2, 3, 4):
            raise ValueError("invalid block bitwidth")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and nonnegative")
        length = (count * bitwidth + 7) // 8
        if offset + length > len(wire):
            raise ValueError("truncated packed codes")
        packed = wire[offset:offset + length]
        offset += length
        tail_bits = (count * bitwidth) % 8
        if tail_bits and packed[-1] >> tail_bits:
            raise ValueError("unused tail bits must be zero")
        qmax = (1 << (bitwidth - 1)) - 1
        mask = (1 << bitwidth) - 1
        d = np.empty(count, dtype=np.float32)
        accumulator = available = byte_index = 0
        for coordinate in range(count):
            if available < bitwidth:
                accumulator |= packed[byte_index] << available
                available += 8
                byte_index += 1
            code = accumulator & mask
            accumulator >>= bitwidth
            available -= bitwidth
            if code == mask:
                raise ValueError("unused maximum code is not canonical")
            if scale == 0 and code != qmax:
                raise ValueError("zero scale requires the canonical zero code")
            d[coordinate] = code - qmax
        with np.errstate(over="ignore", invalid="ignore"):
            reconstructed = d * np.float32(scale)
        if not np.isfinite(reconstructed).all():
            raise ValueError("decoded coordinates must be finite float32")
        start = block_index * BLOCK_SIZE
        output[start:start + count] = reconstructed
    if offset != len(wire):
        raise ValueError("trailing bytes are not permitted")
    return torch.from_numpy(output)


def encode(u: torch.Tensor, budget: int, mode: str = "uniform3",
           importance: torch.Tensor | None = None) -> EncodedUpdate:
    """Encode a full-coordinate update within an exact total-byte budget.

    ``mixed_sensitive`` uses mean nonnegative importance per block solely for
    allocation.  The intended importance is a local gradient mean-square proxy,
    not a Fisher estimate.  All-zero importance falls back to unweighted MSE.
    Greedy upgrades maximize positive whole-vector decoded-MSE reduction/byte;
    the finite search does not promise globally optimal quantization/allocation.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer number of bytes")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    values, original_shape = _input_array(u, "u")
    numel = int(values.size)
    if numel > 0xFFFFFFFF:
        raise ValueError("too many coordinates for this wire version")
    importance_values = _importance_array(importance, numel)
    if mode == "mixed_sensitive" and importance_values is None:
        raise ValueError("mixed_sensitive requires coordinate importance")
    minimum = wire_size(numel, 3 if mode == "uniform3" else 2)
    if budget < minimum:
        raise ValueError(f"budget {budget} is below required {minimum} bytes")
    nblocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_lengths = [min(BLOCK_SIZE, numel - i * BLOCK_SIZE) for i in range(nblocks)]
    candidates: list[dict[int, _Candidate]] = []
    for index, count in enumerate(block_lengths):
        block = values[index * BLOCK_SIZE:index * BLOCK_SIZE + count]
        widths = (3,) if mode == "uniform3" else (2, 3, 4)
        candidates.append({width: _quantize(block, width) for width in widths})

    # Normalize all weights by a common constant to avoid numerical overflow;
    # this does not change greedy ordering. Means retain coordinate semantics.
    importance_max = float(np.max(importance_values)) if importance_values is not None else 0.0
    weighted = mode == "mixed_sensitive" and importance_max > 0
    block_weights = np.ones(nblocks, dtype=np.float64)
    if weighted:
        scaled_importance = importance_values / importance_max
        block_weights = np.array([
            np.mean(scaled_importance[i * BLOCK_SIZE:i * BLOCK_SIZE + count])
            for i, count in enumerate(block_lengths)
        ], dtype=np.float64)
    bitwidths = [3 if mode == "uniform3" else 2] * nblocks
    actual_bytes = minimum
    upgrades: list[dict[str, Any]] = []
    if mode != "uniform3":
        while True:
            best_index: int | None = None
            best_ratio = 0.0
            best_gain = 0.0
            best_cost = 0
            for index, width in enumerate(bitwidths):
                if width == 4:
                    continue
                old, new = candidates[index][width], candidates[index][width + 1]
                cost = new.value_bytes - old.value_bytes
                if actual_bytes + cost > budget:
                    continue
                gain = (old.mse - new.mse) * (block_lengths[index] / numel) * float(block_weights[index])
                # A one-coordinate tail can have a zero-byte upgrade. Such an
                # upgrade is still allowed only if it improves distortion.
                ratio = gain / cost if cost else (math.inf if gain > 0 else 0.0)
                if gain > 0 and (best_index is None or ratio > best_ratio):
                    best_index, best_ratio, best_gain, best_cost = index, ratio, gain, cost
            if best_index is None:
                break
            previous = bitwidths[best_index]
            bitwidths[best_index] += 1
            actual_bytes += best_cost
            upgrades.append({"block": best_index, "from": previous, "to": previous + 1,
                             "added_bytes": best_cost, "normalized_mse_gain": best_gain})

    chunks = [HEADER.pack(MAGIC, VERSION, numel, BLOCK_SIZE, nblocks, bytes(40))]
    scales: list[float] = []
    block_mse: list[float] = []
    for index, width in enumerate(bitwidths):
        candidate = candidates[index][width]
        scales.append(candidate.scale)
        block_mse.append(candidate.mse)
        chunks.append(struct.pack("<Bf", width, candidate.scale))
        chunks.append(_pack_codes(candidate.codes, width))
    wire = b"".join(chunks)
    if len(wire) != actual_bytes or len(wire) > budget:
        raise RuntimeError("internal wire accounting mismatch")
    sent_flat = decode(wire)
    input_flat = torch.from_numpy(values)
    residual_flat = input_flat - sent_flat
    reconstructed_sum = sent_flat + residual_flat
    ef_error = (reconstructed_sum.to(torch.float64) - input_flat.to(torch.float64)).abs()
    actual_error = values.astype(np.float64) - sent_flat.numpy().astype(np.float64)
    diagnostics: dict[str, Any] = {
        "mode": mode, "numel": numel, "block_size": BLOCK_SIZE, "nblocks": nblocks,
        "budget_bytes": budget, "wire_bytes": len(wire), "header_bytes": HEADER_SIZE,
        "block_metadata_bytes": 5 * nblocks,
        "packed_value_bytes": len(wire) - HEADER_SIZE - 5 * nblocks,
        "unused_budget_bytes": budget - len(wire), "bitwidths": bitwidths,
        "bitwidth_counts": {width: bitwidths.count(width) for width in (2, 3, 4)},
        "scales": scales, "block_mse": block_mse,
        "mse": float(np.mean(actual_error * actual_error)),
        "weighted_allocation": weighted,
        "zero_importance_fallback": mode == "mixed_sensitive" and importance_max == 0,
        "importance_normalization": importance_max if weighted else 1.0,
        "normalized_block_weights": block_weights.tolist(), "upgrades": upgrades,
        "ef_conservation_max_abs_error": float(ef_error.max()),
        "ef_conservation_l2_error": float(torch.linalg.vector_norm(ef_error)),
        "quantizer_candidates": "maxabs initialization plus four LS/reassignment iterations",
    }
    return EncodedUpdate(sent_flat.reshape(original_shape), residual_flat.reshape(original_shape), wire, diagnostics)
