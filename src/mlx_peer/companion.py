"""USB companion protocol and local credential storage (no cloud services)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct

from .wire import connect_usb, receive_exact

PORT = 49173
MAX_PAYLOAD = 8 * 1024 * 1024
CHUNK = 4 * 1024 * 1024


class PairingRequired(RuntimeError):
    pass


class CompanionClient:
    def __init__(self, serial, token="", timeout=30):
        self.connection = connect_usb(serial, port=PORT, timeout=timeout)
        self.sequence = 0
        self.token = token

    def request(self, operation, payload=b"", **fields):
        if len(payload) > MAX_PAYLOAD:
            raise ValueError("Payload exceeds USB frame limit")
        self.sequence += 1
        header = json.dumps({**fields, "protocol_version": 2, "request_id": self.sequence,
                             "operation": operation, "token": self.token,
                             "payload_bytes": len(payload)}, separators=(",", ":")).encode()
        if len(header) > 16384:
            raise ValueError("USB header is too large")
        try:
            self.connection.sendall(struct.pack("!I", len(header)) + header + payload)
            size, = struct.unpack("!I", receive_exact(self.connection, 4))
            if not 0 < size <= 16384:
                raise ValueError("Invalid USB reply header")
            reply = json.loads(receive_exact(self.connection, size))
            if (reply.get("protocol_version"), reply.get("request_id")) != (2, self.sequence):
                raise ValueError("Unsupported app version or mismatched reply")
            body_size = reply.get("payload_bytes")
            if type(body_size) is not int or not 0 <= body_size <= MAX_PAYLOAD:
                raise ValueError("Invalid USB reply size")
            body = receive_exact(self.connection, body_size)
            if reply.get("ok") is not True:
                if reply.get("error_code") == "pairing_required":
                    raise PairingRequired(reply.get("error", "Pair with your iPhone first"))
                raise RuntimeError(reply.get("error", "iPhone request failed"))
            return reply, body
        except (OSError, ConnectionError, ValueError, PairingRequired):
            self.close()
            raise

    def close(self):
        self.connection.close()


def file_manifest(directory):
    result = {}
    for name in ("config.json", "request.json", "weights.safetensors"):
        file = Path(directory) / name
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result[name] = {"size": file.stat().st_size, "sha256": digest}
    return result


def model_identity(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def transfer(client, directory, progress=lambda *_: None, cancelled=lambda: False):
    """Resume by exact byte offset; phone commits only after all hashes match."""
    directory = Path(directory)
    files = file_manifest(directory)
    identity = model_identity(files)
    reply, _ = client.request("begin", model_id=identity, files=files)
    offsets = reply.get("offsets", {})
    total = sum(f["size"] for f in files.values())
    done = 0
    for name, info in files.items():
        offset = offsets.get(name)
        if type(offset) is not int or not 0 <= offset <= info["size"]:
            raise ValueError("iPhone returned an invalid transfer offset")
        done += offset
        progress(done, total)
        with (directory / name).open("rb") as stream:
            stream.seek(offset)
            while block := stream.read(CHUNK):
                if cancelled():
                    raise InterruptedError("Transfer cancelled. It can resume next time.")
                reply, _ = client.request("chunk", block, model_id=identity, file=name, offset=offset)
                offset += len(block); done += len(block)
                if reply.get("offset") != offset:
                    raise ValueError("iPhone acknowledged an unexpected file position")
                progress(done, total)
    client.request("commit", model_id=identity)
    return identity


class Credentials:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file = self.root / "paired-phones.json"

    def _read(self):
        if not self.file.exists():
            return {}
        if self.file.is_symlink():
            raise ValueError("Pairing store must not be a symbolic link")
        return json.loads(self.file.read_text())

    def get(self, peer_id):
        return self._read().get(peer_id, "")

    def save(self, peer_id, token):
        if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("Invalid pairing token")
        value = self._read(); value[peer_id] = token
        temporary = self.root / (".pairing-" + secrets.token_hex(8))
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.file)
        finally:
            temporary.unlink(missing_ok=True)
