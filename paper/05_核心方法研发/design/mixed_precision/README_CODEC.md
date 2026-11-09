# Fixed-byte full-coordinate codec

`codec.py` is an independent CPU implementation of existing scalar quantization,
greedy block bit allocation, and error-feedback bookkeeping. It does not claim
a new compression mechanism, a globally optimal fit/allocation, or a quantizer
contraction bound. No training, network, authentication, or chain is included.

## API

```python
from codec import encode, decode, wire_size
result = encode(u, budget=54808, mode="uniform3")
result = encode(u, budget=54808, mode="mixed_mse")
result = encode(u, budget=54808, mode="mixed_sensitive", importance=grad_mean_square)
assert len(result.wire) <= 54808
received_flat = decode(result.wire)
```

`u` must be a nonempty finite real floating-point Torch tensor representable as
finite FP32. The update is explicitly converted to detached CPU FP32 before
encoding. The dataclass `EncodedUpdate` contains `sent` and `residual` as CPU FP32
tensors with the original shape, `wire` as bytes, and a JSON-compatible
`diagnostics` dict. `sent` is produced by actually decoding `wire`; residual is
the FP32 subtraction `u_fp32 - sent`. Public `decode` returns a flat tensor: shape
is not transmitted. `wire_size(P,b)` computes the exact uniform-b total length.

`importance` has P finite nonnegative coordinates. The sensitive mode requires
it; its intended source is the local gradient mean-square proxy, not a Fisher
estimate. Its mean within each block weights allocation, without changing the
update, quantizer, or codes available at any bitwidth. A common positive scaling
of all weights does not affect allocation. All-zero importance falls back to
unweighted MSE. Input importance is not clipped.

## Wire, version 1

All multibyte fields are little endian. Header is exactly `struct <8sIIII40s>`:

| Byte offset | Bytes | Field |
| --- | ---: | --- |
| 0 | 8 | magic `MPQCODEC` |
| 8 | 4 | uint32 version = 1 |
| 12 | 4 | uint32 P, strictly positive |
| 16 | 4 | uint32 block size = 256 |
| 20 | 4 | uint32 Nblocks = ceil(P/256) |
| 24 | 40 | reserved, all zero |

Then each continuous coordinate block is encoded independently:

| Field | Bytes |
| --- | ---: |
| bitwidth b, uint8 in {2,3,4} | 1 |
| finite nonnegative FP32 scale | 4 |
| n codes, b bits per code, LSB first within each code/byte | ceil(n*b/8) |

Here n is 256 except the last short block. The high unused bits of each block's
last byte are zero; packing resumes at a new byte for the next block. With
`qmax=2**(b-1)-1`, signed integer d is stored as `code=d+qmax`. The maximum code
`2**b-1` is unused and rejected. A zero scale requires every code to equal qmax.
Decoded values are the actual FP32 product `(code-qmax)*scale`. The decoder
checks magic, version, all metadata, exact total length, reserved bytes,
bitwidths, scales, codes, padding, and finite reconstruction; trailing bytes and
truncated blocks are rejected.

For P=136874: Nblocks=535, with 534 full blocks and n=170 in the last block.
Header + per-block metadata = 64 + 535*5 = 2739 bytes.

| Uniform bitwidth | Packed values | Actual total |
| --- | ---: | ---: |
| 2 | 34219 B | 36958 B |
| 3 | 51328 B | 54067 B |
| 4 | 68437 B | 71176 B |

The comparison budget is `64+8*floor(.05*136874)=54808 B`: Top-K5% has k=6843,
with 4 bytes per index plus 4 bytes per value and the same 64-byte header.
Uniform3 therefore leaves 741 bytes unused. Mixed payloads use the same exact
accounting, including independent short-block byte padding. Byte equality is
not required; byte-budget compliance is.

## Quantizer and allocation

For each block and each available b, initialize scale=maxabs/qmax, assign
`d=round(u/scale).clip(-qmax,qmax)` using ties-to-even rounding, then perform
exactly four least-squares scale updates `(u dot d)/(d dot d)` followed by
reassignment. Initialization and the four updates are the five candidates.
Before evaluating each candidate, roundtrip its scale through FP32 and
recompute d and the actual FP32 decoded q. Retain the candidate with the lowest
actual decoded MSE (first candidate wins exact ties). Dot products and error
statistics use float64 to avoid float32 accumulation overflow. The all-zero
block has scale=0 and all codes=qmax. The finite search is not a proof of a
globally optimal quantizer.

Mixed modes start with all b=2. At each step consider each block's immediate
2->3 or 3->4 upgrade, and choose the fitting upgrade with the largest positive
actual whole-vector MSE decrease per extra byte. The whole-vector contribution
is `n/P * (block_MSE_old-block_MSE_new)`; sensitive allocation multiplies this
by the block's mean importance. Internally a common importance normalization
prevents overflow and preserves ranking. Ties choose the lower block index.
Positive-gain zero-byte upgrades are allowed for very short tails. Stop when no
positive-gain fitting upgrade remains, even when some budget is unused. This
precedence-constrained greedy search is not a global knapsack optimum.

EF conservation uses a FP32 tolerance, since recomputing `sent+residual` can
differ from the input by one ULP. Diagnostics record the maximum absolute and L2
recomposition error. No Top-K contraction claim is applied to quantization.

## Verification

Run the local, training-free fixtures with:

```text
/opt/anaconda3/envs/Tinyproto/bin/python work/mixed_precision/test_codec.py
```

`CODEC_FIXTURES.json` records 11 passing fixtures, including an independent NumPy
bit decoder, 0/1/255/256/257 lengths, actual wire sizes, greedy budgets,
importance behavior, LS scale fitting, malformed wires, dynamic ranges, and a
fixed one-ULP EF witness. The Top-K fixture checks its own k/P L2 bound only.

`benchmark_codec.py` measures encode (including its real decode) and independent
decode at P=136874, with one warmup and seven repeats. The initial CPU benchmark
gave median encode seconds 0.03685 / 0.12331 / 0.12196 for uniform3 / mixed_mse /
mixed_sensitive, and independent decode about 0.013 seconds. Timing is local
measurement, not an end-to-end training or network result.
