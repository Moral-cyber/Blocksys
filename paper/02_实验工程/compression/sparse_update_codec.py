"""Reusable sparse-update codec for the front half of the paper.

The codec implements the method retained after the diagnostic comparison:
global Top-K selection with client-side error feedback.  It deliberately keeps
the packet representation explicit (values, sorted flat indices, and metadata)
so the complete encoded object can later be canonicalized and authenticated by
the watermark/HMAC-SM3 layer.

This module contains no training code.  It is the communication component that
is called after local training and before watermark generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class CodecConfig:
    """Wire-format assumptions used for byte accounting."""

    keep_fraction: float = 0.20
    value_bytes: int = 4
    index_bytes: int = 4
    packet_header_bytes: int = 64

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.value_bytes <= 0 or self.index_bytes <= 0:
            raise ValueError("value_bytes and index_bytes must be positive")
        if self.packet_header_bytes < 0:
            raise ValueError("packet_header_bytes cannot be negative")


@dataclass
class SparsePacket:
    """Decoded representation of one client update packet.

    ``indices`` are sorted flat parameter coordinates.  Sorting is not needed
    for decoding, but makes serialization deterministic for the later
    integrity layer.
    """

    values: torch.Tensor
    indices: torch.Tensor
    num_elements: int
    metadata: Dict[str, Any]

    @property
    def selected_count(self) -> int:
        return int(self.indices.numel())

    @property
    def encoded_bytes(self) -> int:
        return int(self.metadata["encoded_bytes"])


class SparseUpdateCodec:
    r"""Global Top-K sparse encoder with optional error feedback.

    The default path is the paper's retained front-half method:

    .. math:: u_i^t = \Delta_i^t + r_i^{t-1},
              \hat u_i^t = C_K(u_i^t),
              r_i^t = u_i^t - \hat u_i^t.

    The codec does not perform quantization or encryption.  Its output is a
    canonical sparse update candidate; the caller binds the serialized packet
    and its round/client/model metadata to the existing watermark and HMAC-SM3
    pipeline.
    """

    def __init__(self, config: CodecConfig | None = None) -> None:
        self.config = config or CodecConfig()

    @staticmethod
    def empty_residual(reference: torch.Tensor) -> torch.Tensor:
        """Return a residual with the same shape, dtype, and device."""

        return torch.zeros_like(reference)

    @staticmethod
    def reset_if_model_changed(
        residual: torch.Tensor,
        residual_model_version: str,
        model_version: str,
    ) -> torch.Tensor:
        """Clear stale error feedback when the model version changes."""

        if residual_model_version != model_version:
            return torch.zeros_like(residual)
        return residual

    def encode(
        self,
        delta: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        *,
        use_error_feedback: bool = True,
        round_id: Optional[int] = None,
        client_id: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Tuple[SparsePacket, torch.Tensor]:
        """Encode one flattened model update and return ``(packet, residual)``.

        The returned residual is detached from the autograd graph.  Indices are
        sorted before returning, which gives a stable representation for
        canonical serialization and authentication.
        """

        if delta.ndim != 1:
            raise ValueError("delta must be a flattened one-dimensional tensor")
        if not delta.is_floating_point():
            raise TypeError("delta must be a floating-point tensor")
        if residual is None:
            residual = torch.zeros_like(delta)
        if residual.shape != delta.shape:
            raise ValueError("residual and delta must have the same shape")

        candidate = delta + residual if use_error_feedback else delta
        num_elements = int(candidate.numel())
        selected_count = max(
            1, min(num_elements, int(num_elements * self.config.keep_fraction))
        )
        selected = torch.topk(
            candidate.abs(), k=selected_count, sorted=False
        ).indices
        selected = torch.sort(selected).values
        values = candidate.index_select(0, selected).detach().clone()

        sparse = torch.zeros_like(candidate)
        sparse.scatter_(0, selected, values)
        next_residual = (
            (candidate - sparse).detach().clone()
            if use_error_feedback
            else torch.zeros_like(delta)
        )

        payload_bytes = selected_count * (
            self.config.value_bytes + self.config.index_bytes
        )
        encoded_bytes = self.config.packet_header_bytes + payload_bytes
        metadata: Dict[str, Any] = {
            "round_id": round_id,
            "client_id": client_id,
            "model_version": model_version,
            "num_elements": num_elements,
            "selected_count": selected_count,
            "keep_fraction": self.config.keep_fraction,
            "value_bytes": self.config.value_bytes,
            "index_bytes": self.config.index_bytes,
            "packet_header_bytes": self.config.packet_header_bytes,
            "payload_bytes": payload_bytes,
            "encoded_bytes": encoded_bytes,
            "error_feedback": bool(use_error_feedback),
        }
        packet = SparsePacket(
            values=values,
            indices=selected,
            num_elements=num_elements,
            metadata=metadata,
        )
        return packet, next_residual

    @staticmethod
    def decode(packet: SparsePacket, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Reconstruct the sparse update represented by ``packet``."""

        if packet.indices.ndim != 1 or packet.values.ndim != 1:
            raise ValueError("packet values and indices must be one-dimensional")
        if packet.indices.numel() != packet.values.numel():
            raise ValueError("packet values and indices must have equal length")
        if packet.indices.numel() != torch.unique(packet.indices).numel():
            raise ValueError("packet indices must be unique")
        if packet.indices.numel() and (
            int(packet.indices.min()) < 0
            or int(packet.indices.max()) >= packet.num_elements
        ):
            raise ValueError("packet index is outside the declared model length")
        update = torch.zeros(packet.num_elements, dtype=dtype, device=device)
        update.scatter_(
            0,
            packet.indices.to(device=device, dtype=torch.long),
            packet.values.to(device=device, dtype=dtype),
        )
        return update

