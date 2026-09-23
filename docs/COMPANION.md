# Mac + iPhone companion preview

MLX Peer 0.2 adds a native Mac app and an iPhone companion. Select a model already on your Mac; the app prepares each device's assigned weights, transfers the iPhone share over USB, checks its SHA-256 hashes, and runs a short conversation across both devices.

This is a developer preview. The Mac download includes Python, MLX, and its other runtime dependencies. No separate Python installation, Terminal server, manual model copying, cloud account, or model download is required to use that app. The iPhone app currently requires an Xcode development installation; there is no App Store or public TestFlight build yet.

## Requirements

- An Apple Silicon Mac running macOS 14 or later. Intel Macs are not supported.
- A compatible physical iPhone running iOS 17 or later, with enough free RAM and storage. The physical validation device is an iPhone 16; other models have not been validated.
- A USB data cable supported by your devices. USB-C or Lightning depends on the iPhone; a charging-only cable will not work.
- A complete, unquantized dense Qwen2 / Qwen2.5 model folder: `config.json`, `tokenizer.json`, tokenizer configuration, and `.safetensors` weights (or their index). Start with a small model such as Qwen2.5-0.5B. Obtain it separately under its own license.

GGUF, quantized checkpoints, Qwen3, hybrid models, adapters, image/audio models, and arbitrary architectures are not supported by this companion release. BF16 / FP32 input is converted to FP16. The original model stays unchanged. Prepared copies use additional disk space on the Mac and iPhone.

## Install and pair

1. Download the Apple Silicon Mac ZIP from the [GitHub releases page](https://github.com/samuelreyes982/mlx-peer/releases), extract it, and move **MLX Peer.app** to Applications.
2. Install the iPhone app from `ios/App/MLXPeerApp.xcodeproj` using Xcode, your signing team, and your own unique bundle identifier. See the [physical-device setup](../ios/README.md#run-on-a-physical-iphone). A signing certificate and provisioning profile are not included in the repository.
3. Open MLX Peer on your iPhone. Connect its cable to the Mac, unlock it, and approve iOS's **Trust This Computer** prompt if shown.
4. Open the Mac app. Select your USB iPhone and enter the six-digit code shown on the phone. Click **Connect iPhone**. The code expires after five minutes; stop and restart sharing to get a fresh code.
5. Click **Choose model folder…**, then **Load on both devices**. Wait for preparation, transfer, and verification to finish.
6. Type a short prompt in the Mac chat window. Keep the iPhone app open, the phone unlocked, and the cable connected during inference.

The Mac preview is ad-hoc signed, **not Developer ID signed or notarized**. macOS may block a downloaded copy. That distribution step remains unfinished; do not treat this as an ordinary notarized consumer installer. The fully documented source build is available for developers. No command in this project disables Gatekeeper or removes quarantine.

On later sessions, open the iPhone app, connect the cable, and leave the Mac's pairing-code box empty. Select the same model folder and load it again. Existing prepared files are reused; completed phone files are rechecked without being retransmitted. Pairing is remembered until you forget the Mac on the phone.

## Controls and limits

- **Stop** on Mac cancels generation or the next bounded transfer/preparation step. Inference already running on the device finishes its current step. **New chat** clears the visible conversation.
- **Stop sharing** on iPhone closes the USB session. Moving the app into the background also stops sharing. Reopening the app may require tapping **Start sharing**.
- Stop sharing before using **Pair a new Mac** or **Remove saved models** on iPhone. Pairing a new Mac revokes the previous Mac's token. Removing saved phone models does not delete your original Mac checkpoint.
- The UI uses a 512-token context and up to 128 output tokens per turn. Shorten a prompt or start a new chat if the conversation exceeds the context budget.
- Automatic partitioning assigns at most four phone layers, with a conservative 512 MiB weight budget and additional memory reserve. The Mac also rejects models exceeding its conservative memory estimate. These are preview safeguards, not a claim that every model within those sizes works.
- Sharing pauses with an error if iOS reports serious/critical heat. Allow the phone to cool before trying again. Charging and inference both consume energy; no net energy, emissions, hardware, cobalt, or e-waste reduction has been measured.
- There is no background phone compute, Wi-Fi mode, cloud fallback, automatic update service, or performance/capacity guarantee.

## Privacy and protocol

The iPhone listener binds to `127.0.0.1:49173`, reached through Apple's USB device transport. It is not advertised on your Wi-Fi network. The legacy experiment server uses a separate port and is not started in normal companion mode.

The first pairing uses the short-lived phone code (five attempts maximum), then a random 256-bit token saved privately on both devices. The app supports one paired Mac at a time. File and activation frames are bounded; imports use fixed filenames, content-based identities, exact resume offsets, and SHA-256 verification before loading. USB protocol v2 is authenticated but does not add application-level encryption; its threat model assumes a trusted Mac, unlocked iPhone, and trusted local USB connection.

Prompts stay in Mac memory during the session and are not written to chat-history files. The most recently selected model folder is remembered in local app preferences. No application telemetry is sent. Models, credentials, and prepared files are local:

- Mac: `~/Library/Application Support/MLX Peer/` (`paired-phones.json` and `Models/`). Quit the Mac app before removing that folder to clear its pairing and prepared models.
- iPhone: the app's private Application Support directory. Pairing and model storage are excluded from iCloud backup.

## Build the Mac app

On Apple Silicon with Python 3.12 and Xcode command-line tools:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
python scripts/build_macos.py
```

The result is `dist/MLX Peer.app` and a versioned ZIP. The build freezes its Python engine, includes MLX's Metal library and dependency license notices, compiles the native SwiftUI window, and verifies an ad-hoc signature. Developer ID signing, hardened-runtime configuration, notarization, and clean-machine distribution validation are separate release work.

The iPhone project requires Swift 6.3 / the tested Xcode toolchain and the pinned MLX Swift dependency. Review package build plugins when Xcode asks. The MLX `CudaBuild` plugin does no CUDA work on macOS. `--companion-test-report` is development instrumentation compiled only with `DEBUG`; it writes a short-lived pairing report to Documents for physical-device tests. Release builds do not include that report path.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| No USB iPhone listed | Check the cable, unlock the phone, and complete the system trust prompt. |
| Cannot reach companion | Open MLX Peer on iPhone and tap Start sharing. Keep it in the foreground. |
| Code rejected/expired | Stop and restart sharing. If already paired, leave the code blank or explicitly forget the previous Mac first. |
| Interrupted transfer | Reconnect and load the same model. The next transfer resumes from saved byte offsets. |
| Checksum failure | Stop sharing and remove saved phone models, then retry the transfer. |
| Unsupported model | Use the complete supported Qwen2/Qwen2.5 folder with unquantized safetensors and tokenizer files. |
| Long-conversation warning | Start a new chat or shorten the prompt. |
| Memory/thermal warning | Choose a smaller model, close other heavy apps, or let the phone cool. |

## Still required for public consumer distribution

Developer ID signing and notarization for the Mac; iPhone icons, distribution signing, archive validation, App Store Connect setup, privacy disclosures, TestFlight testing, and App Review. A GitHub download does not install the iPhone app automatically or replace Apple's distribution requirements.
