# ----------------------------------------------------------------------
# 回环 HTTP 学习服务 + 链上绑定 + 回执验证
#
# 原始位置：paper/05_核心方法研发/scripts/chain_integration/network_service.py
# 关键函数：
#   binding()        —— 第 75 行，把上传包映射成 8 个上链字段（x1 导出的就是这些值）
#   LedgerClient     —— 登记、读回、校验交易与确认深度
#   LearningService  —— 认证、写入、签发纳入回执
#   verify_receipt() —— 客户端侧校验回执（会话、挑战、上下文、模型摘要）
# 论文对应：§6.1–6.3、§7.2–7.3。
# ----------------------------------------------------------------------

"""Loopback HTTP learning adapter with authenticated incorporation receipts.

The EVM registry binds upload identities; SQLite remains the trusted model
incorporation authority. These are different decisions. Fixed full participation
stops safely in OPEN when a client is absent, retaining pending and old EF.
"""
from __future__ import annotations

import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import secrets
import threading
import time
from urllib.parse import quote, urlsplit

DOMAIN = b"CCF-LEARNING-INCORPORATION-RECEIPT-V1\x00"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def identifier(value):
    return hashlib.sha256(value.encode()).hexdigest()


def plain_hex(value):
    return str(value).lower().removeprefix("0x")


class TransportError(RuntimeError):
    pass


class CountingConnection(http.client.HTTPConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_bytes = 0

    def send(self, data):
        self.sent_bytes += len(data)
        return super().send(data)


def http_request(base, method, path, body=b"", headers=None, metrics=None, label=None):
    address = urlsplit(base)
    if address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Only explicit loopback HTTP endpoints are allowed")
    connection = CountingConnection(address.hostname, address.port, timeout=90)
    started = time.perf_counter()
    response_body, status, failure = b"", None, None
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        status = response.status
        response_body = response.read()
        return status, response_body
    except (OSError, http.client.HTTPException) as error:
        failure = type(error).__name__
        raise TransportError(f"Loopback request failed: {failure}") from error
    finally:
        if metrics is not None:
            metrics.append({"label": label or path, "method": method, "path": path,
                            "request_body_bytes": len(body), "request_http_bytes_sent": connection.sent_bytes,
                            "response_body_bytes_received": len(response_body), "status": status,
                            "transport_error": failure, "seconds": time.perf_counter() - started})
        connection.close()


def binding(protocol, envelope):
    packet = envelope.packet
    return {"record_id": packet.record_id, "digest": envelope.digest,
            "activity_hash": identifier(packet.activity), "client_hash": identifier(packet.client),
            "round_id": packet.round_id, "nonce_hash": identifier(packet.nonce),
            "base_model_hash": packet.base_model_hash, "layout_hash": identifier(packet.layout),
            "codec_spec_sha256": packet.codec_spec_sha256}


class LedgerClient:
    def __init__(self, url, chain_id, contract_address, client_addresses):
        self.url, self.chain_id = url, int(chain_id)
        self.contract_address = plain_hex(contract_address)
        self.uploaders = {identifier(name): plain_hex(address) for name, address in client_addresses.items()}
        self.metrics = []
        self.verified = []

    def request(self, method, path, value=None, label=None):
        body = b"" if value is None else canonical(value)
        status, result = http_request(self.url, method, path, body,
                    {"Content-Type": "application/json"}, self.metrics, label)
        parsed = json.loads(result)
        return status, parsed

    def verify(self, expected, result):
        if (int(result.get("status", 0)) != 1 or
                int(result.get("chain_id", -1)) != self.chain_id or
                plain_hex(result.get("contract_address")) != self.contract_address or
                result.get("canonical") is not True or
                int(result.get("confirmation_depth", -1)) < 2 or
                int(result.get("head_block_number", -1)) - int(result.get("block_number", -1)) < 2 or
                plain_hex(result.get("uploader")) != self.uploaders.get(expected["client_hash"]) or
                not result.get("tx_hash") or not result.get("block_hash")):
            raise ValueError("Ledger receipt lacks expected successful canonical-depth evidence")
        actual = result.get("readback", {})
        for name, value in expected.items():
            observed = actual.get(name)
            if name == "round_id":
                match = int(observed) == value
            elif name == "record_id":
                match = observed == value
            else:
                match = plain_hex(observed) == plain_hex(value)
            if not match:
                raise ValueError(f"Ledger readback mismatch: {name}")
        return result

    def ensure(self, expected):
        status, result = self.request("GET", "/record?record_id=" + quote(expected["record_id"], safe=""),
                                     label="chain_record_readback")
        if status == 404:
            status, result = self.request("POST", "/register", expected, "chain_register")
        if status != 200:
            raise ValueError("Ledger registration/readback failed")
        self.verify(expected, result)
        self.verified.append(result)
        return result

    def readback(self, expected):
        status, result = self.request("GET", "/record?record_id=" + quote(expected["record_id"], safe=""),
                                     label="chain_recovery_readback")
        if status != 200:
            raise ValueError("A persisted local record has no confirmed chain registration")
        return self.verify(expected, result)


def receipt_context(envelope, session, challenge):
    packet = envelope.packet
    return {"version": 1, "session": session, "challenge": challenge,
            "activity": packet.activity, "client": packet.client, "round_id": packet.round_id,
            "record_id": packet.record_id, "nonce": packet.nonce,
            "base_model_hash": packet.base_model_hash, "layout": packet.layout,
            "codec": packet.codec, "codec_spec_sha256": packet.codec_spec_sha256,
            "digest": envelope.digest}


def verify_receipt(protocol, raw, envelope, key, session, challenge, ledger):
    """Return a protocol Receipt only after MAC, context, freshness and chain checks."""
    try:
        signed = json.loads(raw)
        body, tag = signed["receipt"], bytes.fromhex(signed["hmac_sm3"])
        expected_tag = hmac.new(key, DOMAIN + canonical(body), "sm3").digest()
        if not hmac.compare_digest(tag, expected_tag):
            raise protocol.AuthenticationError("Receipt HMAC-SM3 mismatch")
        for name, value in receipt_context(envelope, session, challenge).items():
            if body.get(name) != value:
                raise protocol.AuthenticationError("Receipt freshness/context mismatch: " + name)
        state = body.get("state")
        if state not in ("UNKNOWN", protocol.REGISTERED, protocol.ACCEPTED, protocol.INCORPORATED):
            raise protocol.AuthenticationError("Invalid authenticated receipt state")
        if state != "UNKNOWN":
            ledger.verify(binding(protocol, envelope), body["chain"])
        if state == protocol.INCORPORATED:
            result_hash = body.get("result_model_hash")
            if not isinstance(result_hash, str) or len(result_hash) != 64:
                raise protocol.AuthenticationError("Incorporation receipt lacks a result model hash")
        elif body.get("result_model_hash") is not None:
            raise protocol.AuthenticationError("Non-incorporated receipt has an aggregate result")
        return protocol.Receipt(envelope.packet.record_id, state,
                                envelope.digest if state != "UNKNOWN" else None)
    except protocol.AuthenticationError:
        raise
    except (KeyError, ValueError, TypeError) as error:
        raise protocol.AuthenticationError("Malformed or invalid authenticated receipt") from error


class LearningService:
    def __init__(self, protocol, path, keys, codecs, ledger, session):
        self.protocol, self.keys, self.ledger, self.session = protocol, keys, ledger, session
        self.server = protocol.LocalServer(path, keys, codecs)
        self.lock = threading.RLock()
        self.events, self.responses = [], []
        self.server_http, self.thread = None, None
        with self.server.connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS network_chain_receipts "
                               "(record_id TEXT PRIMARY KEY, receipt TEXT NOT NULL)")

    def validate(self, envelope, allow_unknown=True):
        packet = self.server._authenticate(envelope)
        with self.server.connection() as connection:
            existing = connection.execute("SELECT * FROM records WHERE record_id=?", (packet.record_id,)).fetchone()
            if existing:
                self.server._same_record(existing, envelope)
            config = connection.execute("SELECT * FROM activity WHERE id=?", (packet.activity,)).fetchone()
            round_row = connection.execute("SELECT * FROM rounds WHERE activity=? AND round_id=?",
                                          (packet.activity, packet.round_id)).fetchone()
            codec = self.server.codecs.get(packet.codec)
            if (config is None or round_row is None or codec is None or
                    packet.client not in json.loads(config["weights"]) or
                    packet.base_model_hash != round_row["base_hash"] or
                    packet.layout != config["layout"] or packet.codec != config["codec"] or
                    packet.codec_spec_sha256 != config["codec_spec_sha256"] or
                    codec.spec_sha256 != config["codec_spec_sha256"]):
                raise self.protocol.StateError("Packet does not match persisted fixed round context")
            if not existing and (round_row["state"] != "OPEN" or config["model_hash"] != packet.base_model_hash):
                raise self.protocol.StateError("No fresh upload for a closed/stale round")
            conflict = connection.execute("SELECT record_id FROM records WHERE activity=? AND client=? AND nonce=?",
                                          (packet.activity, packet.client, packet.nonce)).fetchone()
            if conflict and conflict["record_id"] != packet.record_id:
                raise self.protocol.CollisionError("Nonce already reserved")
            accepted = connection.execute("SELECT record_id FROM records WHERE activity=? AND round_id=? "
                                         "AND client=? AND state IN ('ACCEPTED','INCORPORATED')",
                                         (packet.activity, packet.round_id, packet.client)).fetchone()
            if accepted and accepted["record_id"] != packet.record_id:
                raise self.protocol.CollisionError("Client already accepted in this fixed round")
        return packet, existing

    def _chain_saved(self, record_id):
        with self.server.connection() as connection:
            row = connection.execute("SELECT receipt FROM network_chain_receipts WHERE record_id=?", (record_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def sign(self, envelope, challenge, chain=None):
        packet = envelope.packet
        state = self.server.query(packet.record_id).state
        result_hash = None
        if state == self.protocol.INCORPORATED:
            with self.server.connection() as connection:
                result_hash = connection.execute("SELECT result_hash FROM rounds WHERE activity=? AND round_id=?",
                                                 (packet.activity, packet.round_id)).fetchone()[0]
        body = receipt_context(envelope, self.session, challenge)
        body.update(state=state, result_model_hash=result_hash,
                    chain=chain if state != "UNKNOWN" else None)
        return canonical({"receipt": body, "hmac_sm3": hmac.new(self.keys[packet.client],
                                  DOMAIN + canonical(body), "sm3").hexdigest()})

    def upload(self, envelope, challenge, drop_after_ledger=False):
        with self.lock:
            packet, existing = self.validate(envelope)
            expected = binding(self.protocol, envelope)
            chain = self.ledger.readback(expected) if existing else self.ledger.ensure(expected)
            self.ledger.verify(expected, chain)
            if drop_after_ledger:
                self.events.append({"event": "dropped_after_chain_before_local_register", "record_id": packet.record_id})
                return None
            with self.server.transaction() as connection:
                connection.execute("INSERT OR REPLACE INTO network_chain_receipts VALUES(?,?)",
                                   (packet.record_id, canonical(chain).decode()))
            self.server.register(envelope)
            accepted = self.server.accept(envelope)
            if accepted.state != self.protocol.INCORPORATED:
                try:
                    final = self.server.finalize(packet.activity, packet.round_id)
                except self.protocol.StateError as error:
                    if str(error) != "fixed full participation has not been accepted":
                        raise
                    self.events.append({"event": "fixed_participation_safe_stop", "round": packet.round_id})
                else:
                    self.events.append({"event": "finalized", "round": packet.round_id,
                                        "newly_finalized": final["newly_finalized"]})
            return self.sign(envelope, challenge, chain)

    def query(self, envelope, challenge):
        with self.lock:
            packet, existing = self.validate(envelope)
            chain = None
            if existing:
                chain = self.ledger.readback(binding(self.protocol, envelope))
            return self.sign(envelope, challenge, chain)

    def start(self):
        service = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *_):
                pass
            def do_POST(self):
                started = time.perf_counter()
                raw = b""
                status, result, dropped = 200, None, False
                try:
                    request_size = int(self.headers.get("Content-Length", 0))
                    if request_size <= 32 or request_size > 2 * 1024 * 1024:
                        raise service.protocol.ProtocolError("Invalid request length")
                    raw = self.rfile.read(request_size)
                    if len(raw) != request_size:
                        raise service.protocol.ProtocolError("Truncated request body")
                    challenge = self.headers.get("X-Receipt-Challenge", "")
                    if len(challenge) != 64 or len(bytes.fromhex(challenge)) != 32:
                        raise service.protocol.ProtocolError("Invalid receipt challenge")
                    envelope = service.protocol.Envelope(raw[:-32], raw[-32:])
                    if self.path == "/upload":
                        result = service.upload(envelope, challenge,
                             self.headers.get("X-Diagnostic-Drop-After-Ledger") == "1")
                    elif self.path == "/query":
                        result = service.query(envelope, challenge)
                    else:
                        raise service.protocol.ProtocolError("Unknown endpoint")
                    dropped = result is None or self.headers.get("X-Diagnostic-Drop-ACK") == "1"
                except (service.protocol.ProtocolError, ValueError) as error:
                    status, result = 400, canonical({"error": type(error).__name__, "message": str(error)})
                response_http_bytes = 0
                if dropped:
                    self.close_connection = True
                else:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(result)))
                    self.send_header("Connection", "close")
                    response_http_bytes = sum(map(len, self._headers_buffer)) + 2 + len(result)
                    self.end_headers()
                    self.wfile.write(result)
                    self.wfile.flush()
                    self.close_connection = True
                service.responses.append({"path": self.path, "request_body_bytes_received": len(raw),
                    "response_body_bytes_sent": 0 if dropped else len(result),
                    "response_http_bytes_sent": response_http_bytes, "dropped_ack": dropped,
                    "status": None if dropped else status, "seconds": time.perf_counter() - started})
        self.server_http = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:" + str(self.server_http.server_port)
        self.thread = threading.Thread(target=self.server_http.serve_forever, daemon=True)
        self.thread.start()
        return self.url

    def stop(self):
        if self.server_http:
            self.server_http.shutdown()
            self.server_http.server_close()
            self.thread.join()
            self.server_http, self.thread = None, None


class NetworkClient:
    def __init__(self, protocol, store, key, session, ledger, url, metrics):
        self.protocol, self.store, self.key, self.session = protocol, store, key, session
        self.ledger, self.url, self.metrics = ledger, url, metrics

    def request(self, envelope, path="/query", *, headers=None, challenge=None):
        challenge = challenge or secrets.token_hex(32)
        request_headers = {"Content-Type": "application/octet-stream", "X-Receipt-Challenge": challenge}
        request_headers.update(headers or {})
        status, raw = http_request(self.url, "POST", path, envelope.wire + envelope.tag,
                                  request_headers, self.metrics, label=path)
        if status != 200:
            return status, raw, None, challenge
        authenticated = verify_receipt(self.protocol, raw, envelope, self.key,
                                       self.session, challenge, self.ledger)
        return status, raw, authenticated, challenge

    def commit(self, envelope, raw, challenge):
        authenticated = verify_receipt(self.protocol, raw, envelope, self.key,
                                       self.session, challenge, self.ledger)
        return self.store.commit_if_incorporated(envelope.packet.record_id, lambda _: authenticated)