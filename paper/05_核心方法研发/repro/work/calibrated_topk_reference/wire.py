"""64-byte TopK wrapper around the existing canonical index/value serializer."""
from dataclasses import dataclass
import math
import struct
import numpy as np
import torch
import common as c

HEADER = struct.Struct('<8sIIII40s')
MAGIC, VERSION, PAIR_SIZE = b'TOPKEF01', 1, 8


@dataclass(frozen=True)
class Encoded:
    wire: bytes
    sent: torch.Tensor
    indices: torch.Tensor


def encode(candidate, budget=c.CAP):
    if (not isinstance(candidate, torch.Tensor) or candidate.ndim != 1
            or candidate.device.type != 'cpu' or candidate.dtype != torch.float32
            or not 1 <= candidate.numel() <= c.PARAMETERS
            or not bool(torch.isfinite(candidate).all())):
        raise ValueError('finite CPU FP32 flat candidate with profile length required')
    if type(budget) is not int or not HEADER.size + PAIR_SIZE <= budget <= c.CAP:
        raise ValueError('integer budget within fixed profile required')
    values = candidate.detach().numpy()
    k = min(len(values), (budget - HEADER.size) // PAIR_SIZE)
    # Stable ordering uses original increasing indices to resolve |u| ties.
    selected = np.argsort(-np.abs(values), kind='stable')[:k]
    indices = torch.from_numpy(np.sort(selected).astype(np.int64))
    payload = c.pairs.canonical_pairs(indices, candidate[indices], len(values))
    wire = HEADER.pack(MAGIC, VERSION, len(values), k, PAIR_SIZE, bytes(40)) + payload
    sent = decode(wire)
    if len(wire) != HEADER.size + PAIR_SIZE * k or len(wire) > budget:
        raise ValueError('actual wire accounting failure')
    return Encoded(wire, sent, indices)


def decode(wire):
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise ValueError('bytes-like wire required')
    wire = bytes(wire)
    if not HEADER.size <= len(wire) <= c.CAP:
        raise ValueError('invalid wire length')
    magic, version, numel, k, pair_size, reserved = HEADER.unpack_from(wire)
    if (magic != MAGIC or version != VERSION or pair_size != PAIR_SIZE or any(reserved)
            or not 1 <= k <= min(numel, c.K) or not 1 <= numel <= c.PARAMETERS
            or len(wire) != HEADER.size + PAIR_SIZE * k):
        raise ValueError('invalid header/count/length or trailing bytes')
    pairs = np.frombuffer(wire, dtype=np.dtype([('index', '<u4'), ('value', '<f4')]), offset=HEADER.size, count=k)
    ids, values = pairs['index'], pairs['value']
    if (np.any(ids >= numel) or (k > 1 and np.any(ids[1:] <= ids[:-1]))
            or not np.isfinite(values).all()):
        raise ValueError('duplicate/unsorted/out-of-range indices or nonfinite FP32 values')
    output = np.zeros(numel, dtype=np.float32)
    output[ids] = values
    return torch.from_numpy(output)
