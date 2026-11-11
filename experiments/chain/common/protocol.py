# ----------------------------------------------------------------------
# 上传包的规范序列化与 HMAC-SM3 认证
#
# 原始位置：paper/05_核心方法研发/scripts/durable_integrity_v2/protocol.py
# 关键类型：
#   Packet    —— 10 个字段：activity/client/round/record_id/nonce/base_model_hash/
#                layout/codec/codec_spec_sha256/payload
#   Envelope  —— wire（Packet.to_bytes() 的字节）+ tag（HMAC-SM3(key, wire)）
#   digest    —— SM3(wire)，即上链的 digest 字段；不需要密钥就能复算
# 论文对应：§6.1、Table 3。
# ----------------------------------------------------------------------

"""Durable LOCAL protocol fixture; not a blockchain or ML benchmark.

SQLite is the only registration/aggregation authority. The trusted server holds
the same HMAC keys as the clients. Authenticating a message does not establish
that its update is honest, useful, private, or robust against poisoning.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import sqlite3
import struct
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


REGISTERED, ACCEPTED, INCORPORATED = "REGISTERED", "ACCEPTED", "INCORPORATED"
MAX_PAYLOAD = 1 << 20
SPEC_SHA256 = MappingProxyType({
    "synthetic-sparse-q8-v1": "90543828505430c4a0df3f2671c9f396015f8c0bf1d6943fa4d0566e4ea2c0ec",
    "frozen-mixed-precision-v1": "73e252614fad7fabc4d4921f526c1ff71e7c0d4c18f4528f443aacb39f689226",
})


class ProtocolError(ValueError):
    pass


class AuthenticationError(ProtocolError):
    pass


class CollisionError(ProtocolError):
    pass


class StateError(ProtocolError):
    pass


def _check_sha256(value: str):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ProtocolError("codec spec identity must be lowercase SHA256 hex")


def frozen_spec_identity(name: str) -> str:
    """Load a versioned wire specification and enforce its pinned byte hash.

    This labels intended semantics; it does not inspect or prove callable
    behavior. A trusted matching codec implementation remains an assumption.
    """
    if name not in SPEC_SHA256:
        raise ProtocolError("unknown frozen codec specification")
    actual = hashlib.sha256((Path(__file__).parent / "specs" / (name + ".json")).read_bytes()).hexdigest()
    if actual != SPEC_SHA256[name]:
        raise ProtocolError("frozen codec specification bytes changed")
    return actual


def sm3(data: bytes) -> str:
    return hashlib.new("sm3", data).hexdigest()


def _vector(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ProtocolError("a nonempty finite vector is required")
    return result


def vector_bytes(values: Sequence[float]) -> bytes:
    values = _vector(values)
    return struct.pack("<" + "d" * len(values), *values)


def bytes_vector(data: bytes) -> tuple[float, ...]:
    if not data or len(data) % 8:
        raise ProtocolError("invalid float64 vector bytes")
    return _vector(struct.unpack("<" + "d" * (len(data) // 8), data))


@dataclass(frozen=True)
class Packet:
    activity: str
    client: str
    round_id: int
    record_id: str
    nonce: str
    base_model_hash: str
    layout: str
    codec: str
    codec_spec_sha256: str
    payload: bytes

    def to_bytes(self) -> bytes:
        if type(self.round_id) is not int or not 0 <= self.round_id < (1 << 63):
            raise ProtocolError("invalid round")
        _check_sha256(self.codec_spec_sha256)
        out = bytearray(b"DURINT02" + struct.pack("<Q", self.round_id))
        for value in (self.activity, self.client, self.record_id, self.nonce,
                      self.base_model_hash, self.layout, self.codec, self.codec_spec_sha256):
            if not isinstance(value, str) or not value:
                raise ProtocolError("binding fields must be nonempty strings")
            encoded = value.encode("utf-8")
            if len(encoded) > 65535:
                raise ProtocolError("binding field is too long")
            out.extend(struct.pack("<H", len(encoded)))
            out.extend(encoded)
        if not isinstance(self.payload, bytes) or len(self.payload) > MAX_PAYLOAD:
            raise ProtocolError("invalid payload")
        out.extend(struct.pack("<Q", len(self.payload)))
        out.extend(self.payload)
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Packet":
        if not isinstance(data, bytes) or len(data) > MAX_PAYLOAD + 8 * 65537 + 24:
            raise ProtocolError("invalid envelope size/type")
        if len(data) < 16 or data[:8] != b"DURINT02":
            raise ProtocolError("invalid envelope version")
        round_id = struct.unpack_from("<Q", data, 8)[0]
        offset, fields = 16, []
        try:
            for _ in range(8):
                length = struct.unpack_from("<H", data, offset)[0]
                offset += 2
                if length == 0 or offset + length > len(data):
                    raise ProtocolError("invalid binding field length")
                fields.append(data[offset:offset + length].decode("utf-8"))
                offset += length
            length = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
        except (struct.error, UnicodeError) as error:
            raise ProtocolError("truncated or invalid envelope") from error
        if length > MAX_PAYLOAD or offset + length != len(data):
            raise ProtocolError("payload length or trailing bytes mismatch")
        packet = cls(fields[0], fields[1], round_id, *fields[2:], data[offset:])
        if packet.to_bytes() != data:
            raise ProtocolError("noncanonical envelope")
        return packet


@dataclass(frozen=True)
class Envelope:
    wire: bytes
    tag: bytes

    @classmethod
    def sign(cls, packet: Packet, key: bytes) -> "Envelope":
        wire = packet.to_bytes()
        return cls(wire, hmac.new(key, wire, "sm3").digest())

    @property
    def digest(self) -> str:
        return sm3(self.wire)

    @property
    def packet(self) -> Packet:
        return Packet.from_bytes(self.wire)


@dataclass(frozen=True)
class Receipt:
    record_id: str
    state: str
    digest: str | None


@dataclass(frozen=True)
class Codec:
    name: str
    spec_sha256: str
    encode: Callable[[Sequence[float]], bytes]
    decode: Callable[[bytes], Sequence[float]]

    def __post_init__(self):
        _check_sha256(self.spec_sha256)
        if not self.name or not callable(self.encode) or not callable(self.decode):
            raise ProtocolError("invalid codec registry entry")


def synthetic_sparse_codec() -> Codec:
    """Deliberately simple synthetic fixture, not a proposed compression method.

    Retain even coordinates and quantize to int8 at fixed scale 0.25. The
    nonzero residual makes pending/commit behavior observable in tiny tests.
    """
    header = struct.Struct("<4sIIfB3x")

    def encode(values: Sequence[float]) -> bytes:
        values = _vector(values)
        pairs = [(i, max(-127, min(127, round(values[i] / 0.25))))
                 for i in range(0, len(values), 2)]
        return header.pack(b"SYQ1", len(values), len(pairs), 0.25, 8) + b"".join(
            struct.pack("<Ib", i, code) for i, code in pairs)

    def decode(payload: bytes) -> tuple[float, ...]:
        if len(payload) < header.size:
            raise ProtocolError("truncated synthetic payload")
        magic, size, count, scale, width = header.unpack_from(payload)
        if (magic != b"SYQ1" or size <= 0 or size > MAX_PAYLOAD // 8 or
                count > size or width != 8 or scale != 0.25 or
                len(payload) != header.size + count * 5 or any(payload[17:20])):
            raise ProtocolError("invalid synthetic payload")
        result, previous = [0.0] * size, -1
        for number in range(count):
            index, code = struct.unpack_from("<Ib", payload, header.size + number * 5)
            if index <= previous or index >= size or code == -128:
                raise ProtocolError("invalid synthetic coordinate/code")
            result[index], previous = code * scale, index
        return tuple(result)

    name = "synthetic-sparse-q8-v1"
    return Codec(name, frozen_spec_identity(name), encode, decode)


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise


class LocalServer(SQLiteStore):
    """Trusted local server + simulated registration store in one SQLite DB."""
    def __init__(self, path: str | Path, keys: Mapping[str, bytes], codecs: Mapping[str, Codec]):
        if any(name != codec.name for name, codec in codecs.items()):
            raise ProtocolError("registry key must equal codec name")
        self.keys, self.codecs = MappingProxyType(dict(keys)), MappingProxyType(dict(codecs))
        super().__init__(path)
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS activity (
                    id TEXT PRIMARY KEY, layout TEXT NOT NULL, codec TEXT NOT NULL,
                    codec_spec_sha256 TEXT NOT NULL,
                    weights TEXT NOT NULL, model BLOB NOT NULL, model_hash TEXT NOT NULL,
                    accumulator BLOB NOT NULL, finalized_count INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS rounds (
                    activity TEXT NOT NULL REFERENCES activity(id), round_id INTEGER NOT NULL,
                    base_hash TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('OPEN','FINALIZED')),
                    result_model BLOB, result_hash TEXT, round_accumulator BLOB,
                    PRIMARY KEY(activity, round_id));
                CREATE TABLE IF NOT EXISTS records (
                    record_id TEXT PRIMARY KEY, activity TEXT NOT NULL, client TEXT NOT NULL,
                    round_id INTEGER NOT NULL, nonce TEXT NOT NULL, wire BLOB NOT NULL,
                    tag BLOB NOT NULL, digest TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('REGISTERED','ACCEPTED','INCORPORATED')),
                    decoded BLOB,
                    UNIQUE(activity, client, nonce));
                CREATE UNIQUE INDEX IF NOT EXISTS one_accepted_client_per_round
                    ON records(activity, round_id, client)
                    WHERE state IN ('ACCEPTED','INCORPORATED');
            """)

    def create_activity(self, activity: str, initial_model: Sequence[float], *,
                        layout: str, codec: str, fixed_weights: Mapping[str, float]) -> str:
        model = _vector(initial_model)
        weights = {client: float(weight) for client, weight in sorted(fixed_weights.items())}
        if (not weights or not set(weights) <= set(self.keys) or
                not all(math.isfinite(w) and w > 0 for w in weights.values()) or
                not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12) or codec not in self.codecs):
            raise ProtocolError("invalid fixed participation weights/codec")
        raw = vector_bytes(model)
        with self.transaction() as conn:
            conn.execute("INSERT INTO activity(id,layout,codec,codec_spec_sha256,weights,model,model_hash,accumulator) "
                         "VALUES(?,?,?,?,?,?,?,?)", (activity, layout, codec, self.codecs[codec].spec_sha256,
                                                   json.dumps(weights, sort_keys=True),
                                                   raw, sm3(raw), vector_bytes([0] * len(model))))
        return sm3(raw)

    def open_round(self, activity: str, round_id: int) -> str:
        if type(round_id) is not int or round_id < 0:
            raise ProtocolError("invalid round")
        with self.transaction() as conn:
            config = conn.execute("SELECT model_hash FROM activity WHERE id=?", (activity,)).fetchone()
            if config is None:
                raise StateError("unknown activity")
            previous = conn.execute("SELECT * FROM rounds WHERE activity=? AND round_id=?",
                                    (activity, round_id)).fetchone()
            if previous:
                if previous["state"] != "OPEN" or previous["base_hash"] != config["model_hash"]:
                    raise StateError("round already finalized or stale")
                return previous["base_hash"]
            if conn.execute("SELECT 1 FROM rounds WHERE activity=? AND state='OPEN'", (activity,)).fetchone():
                raise StateError("another round is open")
            latest = conn.execute("SELECT MAX(round_id) FROM rounds WHERE activity=?", (activity,)).fetchone()[0]
            if latest is not None and round_id <= latest:
                raise StateError("round must advance monotonically")
            conn.execute("INSERT INTO rounds(activity,round_id,base_hash,state) VALUES(?,?,?,'OPEN')",
                         (activity, round_id, config["model_hash"]))
            return config["model_hash"]

    def _authenticate(self, envelope: Envelope) -> Packet:
        packet = envelope.packet
        key = self.keys.get(packet.client)
        if key is None or not hmac.compare_digest(
                hmac.new(key, envelope.wire, "sm3").digest(), envelope.tag):
            raise AuthenticationError("HMAC-SM3 mismatch or unknown client")
        return packet

    @staticmethod
    def _same_record(row, envelope: Envelope):
        if row["wire"] != envelope.wire or not hmac.compare_digest(row["tag"], envelope.tag):
            raise CollisionError("RecordID is already bound to different bytes")

    def register(self, envelope: Envelope) -> Receipt:
        packet = self._authenticate(envelope)
        with self.transaction() as conn:
            previous = conn.execute("SELECT * FROM records WHERE record_id=?", (packet.record_id,)).fetchone()
            if previous:
                self._same_record(previous, envelope)
                return Receipt(packet.record_id, previous["state"], previous["digest"])
            try:
                conn.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,NULL)",
                             (packet.record_id, packet.activity, packet.client, packet.round_id, packet.nonce,
                              envelope.wire, envelope.tag, envelope.digest, REGISTERED))
            except sqlite3.IntegrityError as error:
                raise CollisionError("nonce already reserved for this activity/client") from error
            return Receipt(packet.record_id, REGISTERED, envelope.digest)

    def accept(self, envelope: Envelope) -> Receipt:
        packet = self._authenticate(envelope)
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM records WHERE record_id=?", (packet.record_id,)).fetchone()
            if row is None:
                raise StateError("record must be registered before acceptance")
            self._same_record(row, envelope)
            config = conn.execute("SELECT * FROM activity WHERE id=?", (packet.activity,)).fetchone()
            local_codec = self.codecs.get(packet.codec)
            # All three identities must match before any decode or idempotent
            # accept. The persisted activity pins semantics across reopen.
            if (config is None or local_codec is None or packet.codec != config["codec"] or
                    packet.codec_spec_sha256 != config["codec_spec_sha256"] or
                    local_codec.spec_sha256 != config["codec_spec_sha256"]):
                raise StateError("codec specification identity/configuration mismatch")
            if row["state"] in (ACCEPTED, INCORPORATED):
                return Receipt(packet.record_id, row["state"], row["digest"])
            round_row = conn.execute("SELECT * FROM rounds WHERE activity=? AND round_id=?",
                                     (packet.activity, packet.round_id)).fetchone()
            if config is None or round_row is None or round_row["state"] != "OPEN":
                raise StateError("unknown activity/round or closed round")
            if (packet.client not in json.loads(config["weights"]) or
                    packet.base_model_hash != round_row["base_hash"] or
                    packet.base_model_hash != config["model_hash"] or
                    packet.layout != config["layout"] or packet.codec != config["codec"]):
                raise StateError("packet does not match the fixed round context")
            decoded = _vector(local_codec.decode(packet.payload))
            if len(decoded) != len(bytes_vector(config["model"])):
                raise ProtocolError("decoded layout size mismatch")
            try:
                conn.execute("UPDATE records SET state='ACCEPTED',decoded=? WHERE record_id=?",
                             (vector_bytes(decoded), packet.record_id))
            except sqlite3.IntegrityError as error:
                raise CollisionError("client already has an accepted contribution in this round") from error
            return Receipt(packet.record_id, ACCEPTED, envelope.digest)

    def finalize(self, activity: str, round_id: int, *, after_model_write: Callable | None = None) -> dict:
        """One local SQL transaction commits model + accumulator + record states.

        after_model_write is a test-only crash injection, inside the transaction.
        A duplicate finalization returns its stored round result, even after later
        rounds change the global model. There is no distributed atomicity claim.
        """
        with self.transaction() as conn:
            round_row = conn.execute("SELECT * FROM rounds WHERE activity=? AND round_id=?",
                                     (activity, round_id)).fetchone()
            if round_row is None:
                raise StateError("unknown round")
            if round_row["state"] == "FINALIZED":
                return {"model": bytes_vector(round_row["result_model"]),
                        "model_hash": round_row["result_hash"], "newly_finalized": False}
            config = conn.execute("SELECT * FROM activity WHERE id=?", (activity,)).fetchone()
            rows = conn.execute("SELECT * FROM records WHERE activity=? AND round_id=? AND state='ACCEPTED' "
                                "ORDER BY client", (activity, round_id)).fetchall()
            weights = json.loads(config["weights"])
            if {row["client"] for row in rows} != set(weights):
                raise StateError("fixed full participation has not been accepted")
            if config["model_hash"] != round_row["base_hash"]:
                raise StateError("stale base model")
            model, accumulator = bytes_vector(config["model"]), bytes_vector(config["accumulator"])
            updates = [(weights[row["client"]], bytes_vector(row["decoded"])) for row in rows]
            delta = tuple(math.fsum(weight * values[i] for weight, values in updates) for i in range(len(model)))
            new_model = vector_bytes([a + b for a, b in zip(model, delta)])
            new_accumulator = vector_bytes([a + b for a, b in zip(accumulator, delta)])
            new_hash = sm3(new_model)
            conn.execute("UPDATE activity SET model=?,model_hash=?,accumulator=?,finalized_count=finalized_count+1 "
                         "WHERE id=?", (new_model, new_hash, new_accumulator, activity))
            if after_model_write is not None:
                after_model_write()
            conn.execute("UPDATE records SET state='INCORPORATED' WHERE activity=? AND round_id=? AND state='ACCEPTED'",
                         (activity, round_id))
            conn.execute("UPDATE rounds SET state='FINALIZED',result_model=?,result_hash=?,round_accumulator=? "
                         "WHERE activity=? AND round_id=?", (new_model, new_hash, vector_bytes(delta), activity, round_id))
            return {"model": bytes_vector(new_model), "model_hash": new_hash, "newly_finalized": True}

    def query(self, record_id: str) -> Receipt:
        with self.connection() as conn:
            row = conn.execute("SELECT state,digest FROM records WHERE record_id=?", (record_id,)).fetchone()
            return Receipt(record_id, row["state"], row["digest"]) if row else Receipt(record_id, "UNKNOWN", None)

    def snapshot(self, activity: str) -> dict:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM activity WHERE id=?", (activity,)).fetchone()
            if row is None:
                raise StateError("unknown activity")
            return {"model": bytes_vector(row["model"]), "accumulator": bytes_vector(row["accumulator"]),
                    "model_hash": row["model_hash"], "finalized_count": row["finalized_count"]}


class ClientStore(SQLiteStore):
    """Durable EF pending/commit, one outstanding contribution per client.

    Base-model hashes identify rounds; they do not change the EF coordinate
    schema. The fixed layout ID is checked separately and does not reset EF.
    """
    def __init__(self, path: str | Path, *, activity: str, client: str, key: bytes,
                 layout: str, initial_residual: Sequence[float]):
        self.activity, self.client, self.key, self.layout = activity, client, key, layout
        super().__init__(path)
        initial = vector_bytes(initial_residual)
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ef (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), activity TEXT NOT NULL,
                    client TEXT NOT NULL, layout TEXT NOT NULL, residual BLOB NOT NULL,
                    generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS pending (
                    record_id TEXT PRIMARY KEY, nonce TEXT NOT NULL UNIQUE, wire BLOB NOT NULL,
                    tag BLOB NOT NULL, digest TEXT NOT NULL, rnext BLOB NOT NULL,
                    generation INTEGER NOT NULL, state TEXT NOT NULL CHECK(state IN ('PENDING','COMMITTED')));
                CREATE UNIQUE INDEX IF NOT EXISTS one_pending ON pending(state) WHERE state='PENDING';
            """)
        with self.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO ef VALUES(1,?,?,?,?,0)", (activity, client, layout, initial))
            row = conn.execute("SELECT * FROM ef").fetchone()
            if (row["activity"], row["client"], row["layout"]) != (activity, client, layout):
                raise StateError("client database identity/layout mismatch")
            if len(row["residual"]) != len(initial):
                raise StateError("client database residual dimension mismatch")

    def prepare(self, *, round_id: int, record_id: str, nonce: str, base_model_hash: str,
                codec: Codec, update: Sequence[float]) -> Envelope:
        with self.transaction() as conn:
            previous = conn.execute("SELECT * FROM pending WHERE record_id=? OR state='PENDING' ORDER BY state DESC",
                                    (record_id,)).fetchone()
            if previous:
                envelope = Envelope(previous["wire"], previous["tag"])
                packet = envelope.packet
                expected = (round_id, record_id, nonce, base_model_hash, codec.name, codec.spec_sha256)
                if (packet.round_id, packet.record_id, packet.nonce, packet.base_model_hash,
                        packet.codec, packet.codec_spec_sha256) != expected:
                    raise StateError("unresolved pending contribution; query/resend its exact bytes")
                # Do not regenerate a contribution on retries, even if update changed.
                return envelope
            if conn.execute("SELECT 1 FROM pending WHERE nonce=?", (nonce,)).fetchone():
                raise CollisionError("client nonce already used; do not generate a new contribution")
            state = conn.execute("SELECT * FROM ef").fetchone()
            residual, update = bytes_vector(state["residual"]), _vector(update)
            if len(update) != len(residual):
                raise ProtocolError("update layout size mismatch")
            effective = _vector([a + b for a, b in zip(update, residual)])
            payload = codec.encode(effective)
            sent = _vector(codec.decode(payload))
            if len(sent) != len(effective):
                raise ProtocolError("codec changed the coordinate schema")
            rnext = vector_bytes([a - b for a, b in zip(effective, sent)])
            envelope = Envelope.sign(Packet(self.activity, self.client, round_id, record_id, nonce,
                                            base_model_hash, self.layout, codec.name, codec.spec_sha256, payload), self.key)
            try:
                conn.execute("INSERT INTO pending VALUES(?,?,?,?,?,?,?,'PENDING')",
                             (record_id, nonce, envelope.wire, envelope.tag, envelope.digest, rnext, state["generation"]))
            except sqlite3.IntegrityError as error:
                raise CollisionError("client RecordID/nonce already used") from error
            return envelope

    def pending(self) -> Envelope | None:
        with self.connection() as conn:
            row = conn.execute("SELECT wire,tag FROM pending WHERE state='PENDING'").fetchone()
            return Envelope(row["wire"], row["tag"]) if row else None

    def commit_if_incorporated(self, record_id: str, query: Callable[[str], Receipt]) -> bool:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM pending WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            raise StateError("unknown client pending record")
        if row["state"] == "COMMITTED":
            return False
        receipt = query(record_id)
        if receipt.record_id != record_id or receipt.state != INCORPORATED:
            return False
        if receipt.digest is None or not hmac.compare_digest(receipt.digest, row["digest"]):
            raise AuthenticationError("incorporation receipt digest mismatch")
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM pending WHERE record_id=?", (record_id,)).fetchone()
            if row["state"] == "COMMITTED":
                return False
            state = conn.execute("SELECT * FROM ef").fetchone()
            if state["generation"] != row["generation"]:
                raise StateError("EF generation changed while contribution was pending")
            conn.execute("UPDATE ef SET residual=?,generation=generation+1 WHERE singleton=1", (row["rnext"],))
            conn.execute("UPDATE pending SET state='COMMITTED' WHERE record_id=?", (record_id,))
        return True

    def snapshot(self) -> dict:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM ef").fetchone()
            return {"residual": bytes_vector(row["residual"]), "generation": row["generation"],
                    "layout": row["layout"]}