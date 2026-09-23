# App Review notes — prepared draft

MLX Peer is a wired iPhone compute companion for an Apple Silicon Mac. It is not a standalone iPhone chatbot. No account, subscription, in-app purchase, or demo credentials are required. A physical Mac, iPhone, and USB data cable are required to review its core functionality; simulator screenshots only demonstrate the real interface.

1. Download the Apple Silicon Mac companion from https://github.com/samuelreyes982/mlx-peer/releases. Current Mac downloads are developer previews, ad-hoc signed and not notarized. **Resolve Mac Developer ID signing/notarization before public App Store review.**
2. Separately download Qwen/Qwen2.5-0.5B-Instruct from https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct. Put config.json, model.safetensors, and the tokenizer files into one folder. The model is approximately 1 GB. No model is included in the iPhone binary.
3. Connect the iPhone to the Mac with a USB data cable. Unlock it and accept the system Trust This Computer prompt if shown. Keep MLX Peer in the foreground on the iPhone.
4. Start sharing on the phone. In the Mac app, select the phone and enter the displayed six-digit code. Codes expire in five minutes. The pairing persists until revoked.
5. Choose the local model folder on the Mac, then load on both devices. The Mac transfers the phone's model portion automatically. The tested configuration assigns four layers to the iPhone and uses a 512-token context.
6. In the Mac chat, enter: "What is 12 + 7? Answer in one short sentence." The Mac renders the answer; the phone reports computation status. Then test stopping generation, reconnecting, and reusing the saved model without retransferring its weights.
7. Stop sharing on the phone to enable Pair a new Mac and Remove saved models. Backgrounding or locking the phone stops sharing. Privacy, support, and license information are accessible in the iPhone app.

Implementation: the iPhone listens on loopback TCP port 49173. The Mac routes the connection through the system USB multiplexer. The iPhone uses public POSIX networking, MLX/Metal, Foundation and CryptoKit APIs; it does not use a private iOS USB framework. A pairing credential authenticates the Mac. The app's USB protocol has no additional end-to-end encryption. Imported model files are bounded numerical data and configuration, never executable plug-ins or downloaded application code.

Supported companion models are dense Qwen2/Qwen2.5 prepared as FP16. The repository contains separate developer experiments with other models; those are not features of the submitted app. The iPhone app performs no mining, proof-of-work, advertisements, tracking, or background compute. Inference is user initiated on the paired Mac and remains foreground-only on the phone.

Before copying these notes into App Store Connect, insert the final signed Mac release URL and remove the resolved signing blocker. Provide the real developer review contact in App Store Connect. Do not invent a contact number or attach private pairing credentials. A reviewer may request additional hardware or a demonstration; provide it if requested.
