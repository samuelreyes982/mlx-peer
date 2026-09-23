# MLX Peer support

MLX Peer by Samuel Reyes shares supported local-model computation between an Apple Silicon Mac and an iPhone over a USB data cable. Chat takes place in the Mac app. The iPhone app runs the model layers assigned to it; it is not a standalone chatbot.

## Set up

1. Get the Apple Silicon Mac companion from [GitHub Releases](https://github.com/samuelreyes982/mlx-peer/releases). macOS 14 or later is required. Current GitHub builds are developer previews and are ad-hoc signed, not Developer ID signed or notarized. Follow the release's installation notes.
2. Connect your iPhone to the Mac using a USB data cable. Unlock it and accept Apple's **Trust This Computer** prompt if shown.
3. Open MLX Peer on the iPhone and start sharing. Keep the app open and the phone unlocked.
4. Select the connected phone in the Mac app and enter its six-digit pairing code. The code expires after five minutes; restart sharing to generate a fresh code. Pairing is remembered for later sessions.
5. Select a supported local model folder in the Mac app. Start with [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Download its `config.json`, tokenizer files, and `model.safetensors` into one folder before selecting it. The Mac prepares and transfers the assigned layers automatically. No model is bundled with MLX Peer.
6. Load the model and chat from the Mac. Stop generation on the Mac or stop sharing on the iPhone whenever you need to.

The companion currently supports dense Qwen2/Qwen2.5 models prepared as FP16. GGUF, quantized checkpoints, other architectures, and the repository's separate experimental 27B research path are not supported by this app flow. Use a small instruct model for chat. Storage, memory, and temperature can limit what the phone can run.

## Troubleshooting

- **Phone missing:** Confirm the cable carries data, not just power. Unlock the phone, check the Trust prompt, and keep MLX Peer open. Reconnect the cable and refresh devices on the Mac.
- **Expired or rejected code:** Stop and restart sharing. After five wrong attempts, restart sharing. To change the paired Mac, stop sharing and choose **Pair a new Mac**.
- **Connection closed:** The phone stops sharing when backgrounded or locked. Bring it back to the foreground and restart sharing, then reconnect from the Mac.
- **Out of memory, storage, or too warm:** Stop sharing. Let the phone cool. Remove saved models if storage is low, and use a smaller model, fewer phone layers, or a shorter context on the Mac.
- **Unexpected answers:** Small local models make mistakes. Use an instruct checkpoint for chat. Check important output yourself.

USB-C describes a connector, not guaranteed Thunderbolt support. The tested iPhone 16 uses USB 2. MLX Peer does not currently offer Wi-Fi pairing or promise a speed, capacity, energy, or environmental improvement over running a model on the Mac alone.

## Get help

[Open a GitHub issue](https://github.com/samuelreyes982/mlx-peer/issues/new) with the app version, macOS/iOS version, device models, public model name, and the visible error message. Issues are public: do not attach prompts, pairing codes, credentials, personal files, or private model weights.

[Privacy policy](PRIVACY.md) · [Detailed companion guide](COMPANION.md) · [Recorded device validation](COMPANION_VALIDATION.md)
