import json
import socket
import struct
import threading
from unittest.mock import patch

import pytest

from mlx_peer.wire import WireClient, receive_exact, usb_devices, connect_usb


def test_fragmented_reads_and_early_disconnect():
    a, b = socket.socketpair()
    with a, b:
        a.settimeout(1)
        def sender():
            for fragment in (b'a', b'bc', b'def'):
                b.sendall(fragment)
            b.shutdown(socket.SHUT_WR)
        t = threading.Thread(target=sender)
        t.start()
        assert receive_exact(a, 6) == b'abcdef'
        with pytest.raises(ConnectionError):
            receive_exact(a, 1)
        t.join()


@pytest.mark.parametrize('response', [
    {'protocol_version': 1, 'request_id': 999, 'payload_bytes': 0, 'ok': True},
    {'protocol_version': 1, 'request_id': 1, 'payload_bytes': 2**40, 'ok': True},
    {'protocol_version': 1, 'request_id': 1, 'payload_bytes': True, 'ok': True},
    {'protocol_version': 1, 'request_id': 1, 'payload_bytes': 0, 'ok': False, 'error': 'cache mismatch'},
])
def test_bad_response_closes_connection(response):
    a, b = socket.socketpair()
    with b:
        a.settimeout(1)
        encoded = json.dumps(response).encode()
        b.sendall(struct.pack('!I', len(encoded)) + encoded)
        client = WireClient(a, 'a' * 64)
        with pytest.raises((ValueError, RuntimeError)):
            client.request('ping')
        assert a.fileno() == -1


def test_network_devices_cannot_be_selected():
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
    with patch('mlx_peer.wire.mux_socket', return_value=Socket()), patch('mlx_peer.wire.mux_request', return_value={
        'DeviceList': [{'DeviceID': 3, 'Properties': {'ConnectionType': 'Network', 'SerialNumber': 'target'}}]
    }):
        assert usb_devices() == []
        with pytest.raises(ConnectionError):
            connect_usb('target')
