"""CPU codec fixtures; no training or network access.

Run: /opt/anaconda3/envs/Tinyproto/bin/python work/mixed_precision/test_codec.py
The NumPy decoder below reads the documented wire format independently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import unittest

import numpy as np
import torch

from codec import decode, encode


HEADER = struct.Struct("<8sIIII40s")
BLOCK = 256
FIXED_P = 136874
FIXED_BUDGET = 64 + 8 * math.floor(0.05 * FIXED_P)
ERROR_RECORDS: list[dict] = []


def wire_bytes(n: int, widths: int | list[int]) -> int:
    lengths = [min(BLOCK, n - start) for start in range(0, n, BLOCK)]
    if isinstance(widths, int):
        widths = [widths] * len(lengths)
    return HEADER.size + sum(5 + (length * width + 7) // 8
                             for length, width in zip(lengths, widths))


def inspect_wire(wire: bytes) -> list[dict]:
    """Inspect each block without calling any codec helper."""
    magic, version, p, block_size, nblocks, reserved = HEADER.unpack_from(wire)
    assert magic == b"MPQCODEC" and version == 1
    assert p > 0 and block_size == BLOCK and nblocks == (p + BLOCK - 1) // BLOCK
    assert reserved == bytes(40)
    result = []
    pos = HEADER.size
    for index in range(nblocks):
        count = min(BLOCK, p - index * BLOCK)
        bits = wire[pos]
        scale = struct.unpack_from("<f", wire, pos + 1)[0]
        size = (count * bits + 7) // 8
        result.append(dict(index=index, start=index * BLOCK, count=count,
                           bits=bits, scale=scale, offset=pos, data_offset=pos + 5,
                           payload=wire[pos + 5:pos + 5 + size], size=size))
        pos += 5 + size
    assert pos == len(wire)
    return result


def numpy_decode(wire: bytes) -> np.ndarray:
    """Scalar bit extraction independent of implementation packing helpers."""
    blocks = inspect_wire(wire)
    result = []
    for block in blocks:
        bits = block["bits"]
        qmax = (1 << (bits - 1)) - 1
        payload = np.frombuffer(block["payload"], dtype=np.uint8)
        codes = np.empty(block["count"], dtype=np.int32)
        for index in range(block["count"]):
            value = 0
            for bit in range(bits):
                offset = index * bits + bit
                value |= ((int(payload[offset // 8]) >> (offset % 8)) & 1) << bit
            assert value != (1 << bits) - 1
            codes[index] = value
        decoded = (codes - qmax).astype(np.float32) * np.float32(block["scale"])
        result.append(decoded)
    return np.concatenate(result).astype(np.float32, copy=False)


class CodecFixtures(unittest.TestCase):
    def assert_rejected(self, function, *args, **kwargs):
        with self.assertRaises((ValueError, TypeError, OverflowError)):
            function(*args, **kwargs)

    def assert_roundtrip(self, u: torch.Tensor, budget: int, mode: str,
                         importance: torch.Tensor | None = None):
        result = encode(u, budget, mode=mode, importance=importance)
        self.assertIsInstance(result.wire, bytes)
        self.assertEqual(result.sent.dtype, torch.float32)
        self.assertEqual(result.residual.dtype, torch.float32)
        self.assertEqual(result.sent.device.type, "cpu")
        self.assertEqual(result.sent.shape, u.shape)
        self.assertEqual(result.residual.shape, u.shape)
        self.assertLessEqual(len(result.wire), budget)
        decoded = decode(result.wire)
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertEqual(decoded.shape, (u.numel(),))
        self.assertTrue(torch.equal(decoded, result.sent.reshape(-1)))
        reference = numpy_decode(result.wire)
        np.testing.assert_array_equal(decoded.numpy(), reference)
        reference_residual = u.detach().to(device="cpu", dtype=torch.float32) - result.sent
        self.assertTrue(torch.equal(result.residual, reference_residual))
        source = u.detach().to(device="cpu", dtype=torch.float32)
        error = (result.sent + result.residual - source).abs()
        max_error = float(error.max())
        tolerance = 4 * torch.finfo(torch.float32).eps * max(1.0, float(source.abs().max()))
        self.assertLessEqual(max_error, tolerance)
        ERROR_RECORDS.append(dict(n=u.numel(), mode=mode, bytes=len(result.wire),
                                  ef_max_abs_error=max_error, ef_tolerance=tolerance))
        return result

    def test_boundaries_and_independent_numpy_roundtrip(self):
        generator = torch.Generator().manual_seed(20261104)
        for length in (1, 255, 256, 257):
            with self.subTest(length=length):
                u = torch.randn(length, generator=generator)
                result = self.assert_roundtrip(u, wire_bytes(length, 3), "uniform3")
                self.assertEqual(len(result.wire), wire_bytes(length, 3))
                self.assertEqual([b["bits"] for b in inspect_wire(result.wire)],
                                 [3] * ((length + 255) // 256))
                for block in inspect_wire(result.wire):
                    unused = block["size"] * 8 - block["count"] * block["bits"]
                    if unused:
                        self.assertEqual(block["payload"][-1] >> (8 - unused), 0)
        self.assert_rejected(encode, torch.empty(0), 100, mode="uniform3")

    def test_budget_and_fixed_wire_accounting(self):
        self.assertEqual(FIXED_BUDGET, 54808)
        self.assertEqual(wire_bytes(FIXED_P, 3), 54067)
        generator = torch.Generator().manual_seed(81)
        u = torch.randn(FIXED_P, generator=generator)
        uniform = self.assert_roundtrip(u, FIXED_BUDGET, "uniform3")
        self.assertEqual(len(uniform.wire), 54067)
        mixed = self.assert_roundtrip(u, FIXED_BUDGET, "mixed_mse")
        self.assertEqual(len(mixed.wire), wire_bytes(FIXED_P, [b["bits"] for b in inspect_wire(mixed.wire)]))
        self.assertGreaterEqual(len(mixed.wire), wire_bytes(FIXED_P, 2))
        for length in (1, 255, 256, 257):
            u = torch.arange(length, dtype=torch.float32)
            self.assert_rejected(encode, u, wire_bytes(length, 2) - 1, mode="mixed_mse")
            self.assert_rejected(encode, u, wire_bytes(length, 3) - 1, mode="uniform3")
            baseline = self.assert_roundtrip(u, wire_bytes(length, 2), "mixed_mse")
            self.assertEqual([b["bits"] for b in inspect_wire(baseline.wire)],
                             [2] * ((length + 255) // 256))

    def test_topk_only_l2_bound(self):
        # The familiar k/P bound belongs to magnitude Top-K. This fixture does
        # not transfer that bound to either uniform or mixed quantization.
        generator = torch.Generator().manual_seed(6843)
        k = math.floor(0.05 * FIXED_P)
        self.assertEqual(64 + 8 * k, FIXED_BUDGET)
        for values in (torch.ones(FIXED_P, dtype=torch.float64),
                       torch.randn(FIXED_P, generator=generator, dtype=torch.float64)):
            kept = torch.topk(values.abs(), k, sorted=False).indices
            residual = values.clone()
            residual[kept] = 0
            error = float(residual.square().sum())
            bound = (1 - k / FIXED_P) * float(values.square().sum())
            self.assertLessEqual(error, bound + 1e-10 * max(1, bound))

    def test_zero_blocks_do_not_spend_budget(self):
        u = torch.zeros(257)
        result = self.assert_roundtrip(u, wire_bytes(u.numel(), 4), "mixed_mse")
        self.assertEqual(len(result.wire), wire_bytes(u.numel(), 2))
        for block in inspect_wire(result.wire):
            self.assertEqual(block["scale"], 0.0)
            qmax = (1 << (block["bits"] - 1)) - 1
            payload = int.from_bytes(block["payload"], "little")
            for index in range(block["count"]):
                self.assertEqual((payload >> (index * block["bits"])) & ((1 << block["bits"]) - 1), qmax)

    def test_zero_importance_falls_back_to_unweighted(self):
        generator = torch.Generator().manual_seed(391)
        u = torch.randn(1283, generator=generator)
        budget = wire_bytes(u.numel(), 2) + 67
        unweighted = self.assert_roundtrip(u, budget, "mixed_mse")
        weighted = self.assert_roundtrip(u, budget, "mixed_sensitive", torch.zeros_like(u))
        self.assertEqual(unweighted.wire, weighted.wire)

    def test_ef_conservation_allows_one_ulp(self):
        # Clipping after LS makes this witness reconstruct one coordinate one
        # ULP away when sent + residual is evaluated in FP32.
        u = torch.tensor([1.6] + [0.82] * 10 + [0.575] * 245)
        result = self.assert_roundtrip(u, wire_bytes(u.numel(), 2), "mixed_mse")
        error = float((result.sent + result.residual - u).abs().max())
        self.assertGreater(error, 0)
        self.assertEqual(error, torch.finfo(torch.float32).eps)
        self.assertEqual(result.diagnostics["ef_conservation_max_abs_error"], error)

    def test_importance_changes_allocation_only(self):
        generator = torch.Generator().manual_seed(51)
        block = torch.randn(256, generator=generator)
        u = torch.cat([block, block])
        budget = wire_bytes(u.numel(), 2) + 32
        low_first = torch.cat([torch.zeros(256), torch.ones(256)])
        high_first = 1 - low_first
        second = self.assert_roundtrip(u, budget, "mixed_sensitive", low_first)
        first = self.assert_roundtrip(u, budget, "mixed_sensitive", high_first)
        self.assertEqual([b["bits"] for b in inspect_wire(second.wire)], [2, 3])
        self.assertEqual([b["bits"] for b in inspect_wire(first.wire)], [3, 2])
        self.assertTrue(torch.equal(first.sent[:256], second.sent[256:]))
        self.assertTrue(torch.equal(first.sent[256:], second.sent[:256]))

    def test_greedy_budget_marginals_and_determinism(self):
        generator = torch.Generator().manual_seed(998)
        u = torch.cat([torch.randn(256, generator=generator),
                       8 * torch.randn(256, generator=generator),
                       0.125 * torch.randn(17, generator=generator)])
        base = wire_bytes(u.numel(), 2)
        baseline_mse = None
        for extra in (0, 1, 2, 3, 4, 31, 32, 35, 64, 96, 128):
            result = self.assert_roundtrip(u, base + extra, "mixed_mse")
            again = encode(u, base + extra, mode="mixed_mse")
            self.assertEqual(result.wire, again.wire)
            mse = float(((u.to(torch.float64) - result.sent.to(torch.float64)) ** 2).mean())
            # Positive-gain upgrades must improve on the all-2 baseline. Greedy
            # allocation is not claimed to be a globally optimal knapsack solver.
            if baseline_mse is None:
                baseline_mse = mse
            self.assertLessEqual(mse, baseline_mse + 1e-12)
            self.assertEqual(len(result.wire), wire_bytes(u.numel(), [b["bits"] for b in inspect_wire(result.wire)]))

    def test_finite_inputs_shape_and_scale_roundtrip(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.assert_rejected(encode, torch.tensor([value]), 100, mode="uniform3")
            self.assert_rejected(encode, torch.ones(1), 100, mode="mixed_sensitive", importance=torch.tensor([value]))
        self.assert_rejected(encode, torch.ones(2), 100, mode="mixed_sensitive", importance=torch.tensor([1.0, -1.0]))
        self.assert_rejected(encode, torch.ones(2), 100, mode="mixed_sensitive", importance=torch.ones(3))
        generator = torch.Generator().manual_seed(439)
        u = torch.randn((19, 17), generator=generator).T
        self.assertFalse(u.is_contiguous())
        result = self.assert_roundtrip(u, wire_bytes(u.numel(), 3), "uniform3")
        for block in inspect_wire(result.wire):
            self.assertTrue(np.isfinite(block["scale"]))
            self.assertGreaterEqual(block["scale"], 0)
            self.assertEqual(float(np.float32(block["scale"])), block["scale"])
        # A nonzero FP32 scale and LS refinement must remain visible on the wire.
        simple = self.assert_roundtrip(torch.tensor([-1.0, 0.0, 1.0]), 100, "uniform3")
        self.assertAlmostEqual(inspect_wire(simple.wire)[0]["scale"], 1 / 3, places=7)
        self.assertTrue(torch.equal(simple.sent, torch.tensor([-1.0, 0.0, 1.0])))
        # Analytic all-2 fixture: d=(-1,0,1,1), LS scale=(1+.6+1)/3.
        refined = self.assert_roundtrip(torch.tensor([-1.0, 0.2, 0.6, 1.0]),
                                        wire_bytes(4, 2), "mixed_mse")
        self.assertAlmostEqual(inspect_wire(refined.wire)[0]["scale"], 13 / 15, places=7)
        initial_error = torch.tensor([0.0, 0.2, -0.4, 0.0]).square().sum()
        self.assertLess(float((torch.tensor([-1.0, 0.2, 0.6, 1.0]) - refined.sent).square().sum()),
                        float(initial_error))

    def test_decoder_rejects_malformed_wire(self):
        valid = encode(torch.tensor([1.0]), 100, mode="uniform3").wire
        malformed = []
        malformed.extend([b"", valid[:63], valid[:-1], valid + b"\0"])
        for offset, replacement in ((0, b"BADMAGIC"), (8, struct.pack("<I", 2)),
                                    (12, struct.pack("<I", 0)), (16, struct.pack("<I", 255)),
                                    (20, struct.pack("<I", 2)), (24, b"\x01"),
                                    (64, b"\x01"), (64, b"\x05")):
            value = bytearray(valid)
            value[offset:offset + len(replacement)] = replacement
            malformed.append(bytes(value))
        for scale in (-1.0, float("nan"), float("inf"), -float("inf")):
            value = bytearray(valid)
            struct.pack_into("<f", value, 65, scale)
            malformed.append(bytes(value))
        value = bytearray(valid)
        value[69] = 7  # Unused highest code for 3 bits.
        malformed.append(bytes(value))
        value = bytearray(valid)
        value[69] |= 0x80  # High unused bits of a one-coordinate block.
        malformed.append(bytes(value))
        value = bytearray(valid)
        struct.pack_into("<f", value, 65, 0.0)  # Nonzero code with zero scale.
        malformed.append(bytes(value))
        for index, value in enumerate(malformed):
            with self.subTest(malformed=index):
                self.assert_rejected(decode, value)

    def test_random_and_dynamic_range_cases(self):
        generator = torch.Generator().manual_seed(8723)
        for length in (2, 7, 13, 127, 258, 511, 512, 513, 1031):
            for factor in (1e-12, 1.0, 1e12):
                u = torch.randn(length, generator=generator) * factor
                budget = wire_bytes(length, 2) + max(0, (wire_bytes(length, 4) - wire_bytes(length, 2)) // 2)
                self.assert_roundtrip(u, budget, "mixed_mse")
                importance = torch.rand(length, generator=generator)
                self.assert_roundtrip(u, budget, "mixed_sensitive", importance)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CodecFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  errors=len(result.errors), failures=len(result.failures),
                  fixed_p=FIXED_P, fixed_topk5_budget_bytes=FIXED_BUDGET,
                  uniform3_wire_bytes=wire_bytes(FIXED_P, 3),
                  max_ef_abs_error=max((r["ef_max_abs_error"] for r in ERROR_RECORDS), default=0),
                  error_records=ERROR_RECORDS,
                  scope="codec fixtures only; no training; no claim of quantization contraction")
    report_path = Path(__file__).with_name("CODEC_FIXTURES.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "error_records"}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
