# Live USB transport and split-model test

The wired test passed on 2026-09-15 using the M3 Pro Mac and physical iPhone 16 on iOS 27.0. This test used the existing Thunderbolt 4-rated cable as a USB connection. The client selected only a device whose usbmux connection type was `USB`; there was no Wi-Fi fallback.

| Measurement | Result |
| --- | --- |
| Empty application request round trip | Median **0.461 ms**, p95 0.622 ms; 100 requests after 10 warmups |
| 10 KiB echo in each direction | Median **0.963 ms**, p95 1.083 ms; 50 requests |
| Mac → iPhone payload throughput | Median **36.38 MB/s**, eight 8 MiB transfers |
| iPhone → Mac payload throughput | Median **42.74 MB/s**, eight 8 MiB transfers |
| Live phone-stage numerical checks | All three passed |
| Full small-model split logits against Mac reference | All three passed at predeclared FP16 tolerances |
| Live split generation | 24 tokens in **1.06 seconds**, first token after 0.551 seconds |

These are application-level measurements in a foreground debug build. Upload timing includes computing a SHA256 on the phone. The rates include framing, copies, and application work; they are not raw USB signaling rates. The 10 KiB payload size equals one FP16 activation vector of width 5120, although the actual trained-model stage exercised here has width 896.

## What actually ran

The model was **Qwen2.5-0.5B converted to FP16**, with layers 4–7 resident on the iPhone. The Mac coordinator constructed only its assigned layers, embedding and output projection. Each forward pass sent a raw FP16 activation to the phone, which executed its four layers, retained their cache, and returned the result over the same USB connection. The Mac finished the remaining layers and selected the next token.

The Mac shard held 868,766,464 bytes of weights; the phone shard held 119,299,072 bytes. A complete small model was run separately on the Mac to generate reference logits and was released before constructing the split coordinator. This reference-validation procedure is not the proposed larger-than-Mac loading path and is not evidence of a capacity gain.

The three live phone-stage requests reproduced the previously validated prefill and cached continuations. Their maximum absolute errors were 0.0078125, 0.00390625, and 0.00390625, passing the existing `rtol=0.01, atol=0.01` thresholds. The full-model comparisons used thresholds declared in the script before the run (`rtol=0.02, atol=0.05`) and had maximum absolute errors of 0.037109375, 0.033203125, and 0.02734375. This is floating-point agreement within tolerances, not bitwise identity.

The generation prompt was `Question: Why is the sky blue?\nAnswer:`. The 24-token output was:

> The sky is blue because of the scattering of sunlight by tiny particles in the atmosphere.
> A single-select problem: Is the

This is a deliberately short continuation from a base model, not an instruction-following or answer-quality benchmark. No matched Mac-only speed benchmark was collected for this model, so the result establishes live split execution, not a speedup.

## Implementation and failure handling

The phone listener binds only to `127.0.0.1:49172`. The Mac connects through the system usbmux service. A random 32-byte token is pre-provisioned in a private local file and checked on every request; it is not printed in reports. This is a developer transport, not an encrypted general-purpose network service. No public network listener, discovery, or Wi-Fi fallback is enabled.

Requests use a four-byte big-endian JSON-header length, a bounded JSON header, and an optional binary body. Headers are limited to 16 KiB and payloads to 8 MiB. FP16 values are little-endian on these Apple devices. The server handles one client on a serial worker, enforces sequence and cache positions, used 15-second socket timeouts in this recorded small-model test (the current hybrid worker uses 120 seconds), and stops after 600 seconds, 4096 requests, or an authenticated stop request. The benchmark sent `stop` on completion; the server is no longer running.

Physical-device checks confirmed that an incorrect token is rejected, an incorrect cache position is rejected, a partial-frame disconnect can be recovered from, and the next connection starts with reset state and still produces the correct output. Six Python framing/selection tests and three existing Swift validation tests passed. The iOS build and signing succeeded.

The USB framing follows the protocol fields used in [libusbmuxd](https://github.com/libimobiledevice/libusbmuxd/blob/master/src/libusbmuxd.c): `ListDevices`, USB-only selection, and `Connect` with a network-order port value. The implementation uses Python's standard library and the Mac's existing usbmux service.

## Reproduction

Build/install the current app using [the worker instructions](../ios/README.md). Provision a fresh 64-character lowercase hex token as `wire-token.txt` in the same phone fixture folder used for the earlier physical test. Store its Mac copy with mode 0600. Launch the app with `--serve-fixture qwen-probe-20260915-01`; keep it in the foreground. This launch mode is opt-in and does not start during an ordinary app launch.

From the project root, with a fresh output directory:

```sh
.venv/bin/python -u scripts/benchmark_usb.py \
  --serial <USB-device-serial> \
  --token-file <private-token-file> \
  --fixture artifacts/qwen-stage-v3 \
  --tokenizer artifacts/qwen-0.5b-source \
  --output artifacts/usb-probe-repeat
```

The token must match the phone file. This is a fixture-specific development benchmark; it is not yet a general model-serving CLI. It validates hashes before timing, checks payload integrity, saves results incrementally, and stops its server on completion.

The structured report is [USB_PROBE_RESULT.json](USB_PROBE_RESULT.json). Raw results, failure checks, source snapshots, and install/launch records are in `artifacts/usb-probe-20260915/`. Token files stay in ignored private artifacts and must not be published.

This report records the earlier small-model milestone. See the subsequent [Qwen3.8-27B exploratory split result](QWEN38_SPLIT.md), including the outstanding numerical and sustained-capacity checks. The small-model throughput results alone do not establish a larger-model speedup.
