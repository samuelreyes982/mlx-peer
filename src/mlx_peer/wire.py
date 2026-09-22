"""Bounded framing and explicit USB-only usbmux connections (no Wi-Fi fallback)."""

import json
import plistlib
import socket
import struct

MAX_HEADER = 16 * 1024
MAX_PAYLOAD = 8 * 1024 * 1024


def receive_exact(connection, size):
    if not 0 <= size <= MAX_PAYLOAD:
        raise ValueError("Invalid frame size")
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = connection.recv_into(view[offset:])
        if count == 0:
            raise ConnectionError("Peer disconnected mid-frame")
        offset += count
    return bytes(data)


def mux_request(connection, message, tag=1):
    payload = plistlib.dumps({"ClientVersionString": "mlx-peer-wire-probe",
                             "ProgName": "mlx-peer", "kLibUSBMuxVersion": 3, **message})
    connection.sendall(struct.pack("<IIII", 16 + len(payload), 1, 8, tag) + payload)
    length, version, kind, reply_tag = struct.unpack("<IIII", receive_exact(connection, 16))
    if not 16 <= length <= 1024 * 1024 or (version, kind, reply_tag) != (1, 8, tag):
        raise ValueError("Invalid usbmux response")
    return plistlib.loads(receive_exact(connection, length - 16))


def mux_socket(timeout=15):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect("/var/run/usbmuxd")
        return connection
    except BaseException:
        connection.close()
        raise


def usb_devices():
    with mux_socket() as connection:
        result = mux_request(connection, {"MessageType": "ListDevices"})
    return [d for d in result.get("DeviceList", [])
            if d.get("Properties", {}).get("ConnectionType") == "USB"]


def connect_usb(serial, port=49172, timeout=15):
    matches = [d for d in usb_devices() if d["Properties"].get("SerialNumber") == serial]
    if len(matches) != 1:
        raise ConnectionError("Expected one matching physically USB-connected device")
    connection = mux_socket(timeout)
    try:
        reply = mux_request(connection, {"MessageType": "Connect",
            "DeviceID": matches[0]["DeviceID"], "PortNumber": socket.htons(port)})
        if reply.get("Number") != 0:
            raise ConnectionError(f"USB connection rejected: {reply.get('Number')}")
        return connection
    except BaseException:
        connection.close()
        raise


class WireClient:
    def __init__(self, connection, token):
        if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("Expected a 32-byte hex session token")
        self.connection, self.token, self.sequence = connection, token, 0

    def request(self, operation, payload=b"", **fields):
        if len(payload) > MAX_PAYLOAD:
            raise ValueError("Payload exceeds wire limit")
        self.sequence += 1
        header = json.dumps({**fields, "protocol_version": 1, "token": self.token,
            "operation": operation, "request_id": self.sequence,
            "payload_bytes": len(payload)}, separators=(",", ":")).encode()
        if len(header) > MAX_HEADER:
            raise ValueError("Header exceeds wire limit")
        try:
            self.connection.sendall(struct.pack("!I", len(header)) + header + payload)
            length, = struct.unpack("!I", receive_exact(self.connection, 4))
            if not 0 < length <= MAX_HEADER:
                raise ValueError("Invalid reply header length")
            reply = json.loads(receive_exact(self.connection, length))
            if reply.get("request_id") != self.sequence or reply.get("protocol_version") != 1:
                raise ValueError("Response identity mismatch")
            size = reply.get("payload_bytes")
            if type(size) is not int or not 0 <= size <= MAX_PAYLOAD:
                raise ValueError("Invalid reply payload length")
            body = receive_exact(self.connection, size)
            if reply.get("ok") is not True:
                raise RuntimeError(reply.get("error", "Worker error"))
            return reply, body
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()
