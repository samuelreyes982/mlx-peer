import hashlib
import json
import socket
import stat
import struct
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_peer.companion import CompanionClient, Credentials, PairingRequired, file_manifest, model_identity, transfer
from mlx_peer.companion_export import prepare, inspect_model, choose_layers, write_fp16
from mlx_peer.wire import receive_exact
from test_manifest import fixture, read_payloads


def exchange(reply, operation='status'):
    local, remote = socket.socketpair()
    def server():
        with remote:
            length, = struct.unpack('!I', receive_exact(remote, 4))
            request = json.loads(receive_exact(remote, length))
            assert request['operation'] == operation
            raw = json.dumps({'protocol_version': 2, 'request_id': 1, 'payload_bytes': 0, 'ok': True, **reply}).encode()
            remote.sendall(struct.pack('!I', len(raw)) + raw)
    thread = threading.Thread(target=server); thread.start()
    with patch('mlx_peer.companion.connect_usb', return_value=local):
        client = CompanionClient('test')
    return client, thread


@pytest.mark.parametrize('reply,exception', [({'request_id': 3}, ValueError), ({'payload_bytes': -1}, ValueError),
    ({'payload_bytes': 9*1024**2}, ValueError), ({'ok': False, 'error_code': 'pairing_required'}, PairingRequired)])
def test_invalid_reply_closes_connection(reply, exception):
    client, thread = exchange(reply)
    try:
        with pytest.raises(exception): client.request('status')
        assert client.connection.fileno() == -1
    finally: client.close(); thread.join()


class TransferPeer:
    def __init__(self, directory):
        self.files = {name: (directory / name).read_bytes()[:3] for name in file_manifest(directory)}
        self.chunks = 0; self.committed = False
    def request(self, operation, payload=b'', **fields):
        if operation == 'begin':
            self.manifest = fields['files']
            assert model_identity(self.manifest) == fields['model_id']
            return {'offsets': {k: len(v) for k, v in self.files.items()}}, b''
        if operation == 'chunk':
            name = fields['file']; assert len(self.files[name]) == fields['offset']
            self.files[name] += payload; self.chunks += 1
            return {'offset': len(self.files[name])}, b''
        if operation == 'commit':
            for name, data in self.files.items():
                assert len(data) == self.manifest[name]['size']
                assert hashlib.sha256(data).hexdigest() == self.manifest[name]['sha256']
            self.committed = True
            return {}, b''
        raise AssertionError(operation)


def bundle(tmp_path):
    for name in ('config.json', 'request.json', 'weights.safetensors'): (tmp_path / name).write_bytes(b'abcdefghi')
    return tmp_path


def test_transfer_resumes_and_reuses_complete_files(tmp_path):
    directory = bundle(tmp_path); peer = TransferPeer(directory); progress = []
    identity = transfer(peer, directory, lambda done, total: progress.append((done, total)))
    assert peer.committed and peer.chunks == 3 and progress[-1] == (27, 27)
    peer.chunks = 0
    assert transfer(peer, directory) == identity and peer.chunks == 0


def test_cancel_does_not_commit_partial_transfer(tmp_path):
    directory = bundle(tmp_path); peer = TransferPeer(directory)
    with pytest.raises(InterruptedError): transfer(peer, directory, cancelled=lambda: True)
    assert not peer.committed and peer.chunks == 0


def test_transfer_rejects_bad_offset(tmp_path):
    directory = bundle(tmp_path); peer = TransferPeer(directory)
    peer.files['config.json'] = b'x'*100
    with pytest.raises(ValueError, match='offset'): transfer(peer, directory)


def test_credentials_are_private_and_reject_symlink(tmp_path):
    store = Credentials(tmp_path); store.save('phone', 'a'*64)
    assert Credentials(tmp_path).get('phone') == 'a'*64
    assert stat.S_IMODE(store.file.stat().st_mode) == 0o600
    with pytest.raises(ValueError): store.save('phone', 'bad')
    target = tmp_path / 'other.json'; store.file.rename(target); store.file.symlink_to(target)
    with pytest.raises(ValueError, match='symbolic'): store.get('phone')


def test_prepare_fp16_partition_and_cache_invalidation(tmp_path):
    source = tmp_path / 'source'; source.mkdir(); fixture(source); (source / 'tokenizer.json').write_text('{}')
    checkpoint = inspect_model(source)
    directory = prepare(checkpoint, tmp_path / 'cache', 1)
    reconstructed = read_payloads(directory / 'weights.safetensors') | read_payloads(directory / 'mac.safetensors')
    assert reconstructed == read_payloads(source / 'model.safetensors')
    assert prepare(checkpoint, tmp_path / 'cache', 1) == directory
    assert prepare(checkpoint, tmp_path / 'cache', 2) != directory
    assert choose_layers(checkpoint, 2*1024**3) == 1
    with pytest.raises(ValueError, match='memory'): choose_layers(checkpoint, 0)
    with pytest.raises(InterruptedError): prepare(checkpoint, tmp_path / 'cache', 1, cancelled=lambda: True)
    assert not list((tmp_path / 'cache').glob('.preparing-*'))


@pytest.mark.parametrize('dtype', ['BF16', 'F32'])
def test_streaming_conversion_matches_fp16(tmp_path, dtype):
    import numpy as np
    from types import SimpleNamespace
    values = np.array([1.0, -2.5, 0.125], dtype='<f4')
    raw = values.tobytes() if dtype == 'F32' else (values.view('<u4') >> 16).astype('<u2').tobytes()
    source = tmp_path / 'raw'; source.write_bytes(raw)
    tensor = SimpleNamespace(name='weight', shape=[3], nbytes=len(raw), dtype=dtype, source_file=source, absolute_offset=0)
    output = tmp_path / 'converted.safetensors'; write_fp16(output, [tensor])
    assert read_payloads(output)['weight'][1] == values.astype('<f2').tobytes()


def test_nonfinite_weights_rejected(tmp_path):
    from types import SimpleNamespace
    source = tmp_path / 'raw'; source.write_bytes(struct.pack('<f', float('inf')))
    tensor = SimpleNamespace(name='weight', shape=[1], nbytes=4, dtype='F32', source_file=source, absolute_offset=0)
    with pytest.raises(ValueError, match='finite'): write_fp16(tmp_path / 'out', [tensor])
