"""Line-delimited JSON IPC for the native Mac app. Model files never leave USB/local storage."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

from .companion import CompanionClient, Credentials, PairingRequired, transfer
from .companion_export import inspect_model, choose_layers, fp16_size, prepare
from .wire import usb_devices

_output_lock = threading.Lock()


def emit(event, **fields):
    with _output_lock:
        print(json.dumps({"event": event, **fields}), flush=True)


class Engine:
    def __init__(self, support=None):
        self.support = Path(support or Path.home() / "Library/Application Support/MLX Peer")
        self.credentials = Credentials(self.support)
        self.client = None
        self.serial = None
        self.coordinator = None
        self.tokenizer = None
        self.model_id = None
        self.model_path = None
        self.context = 512
        self.cancelled = threading.Event()

    def disconnect(self):
        if self.client:
            try: self.client.request("disconnect")
            except Exception: pass
            self.client.close()
        self.client = None; self.coordinator = None; self.tokenizer = None; self.model_id = None
        if "mlx.core" in sys.modules:
            import mlx.core as mx
            mx.clear_cache()

    def discover(self):
        devices = usb_devices()
        emit("devices", devices=[{"id": d["Properties"]["SerialNumber"], "name": "iPhone connected by USB"} for d in devices])

    def connect(self, serial, code=""):
        self.disconnect()
        try:
            client = CompanionClient(serial)
        except (OSError, ConnectionError) as error:
            raise RuntimeError("Cannot reach the iPhone companion. Open MLX Peer on your iPhone, tap Start sharing, and keep it unlocked with its USB cable connected.") from error
        try:
            hello, _ = client.request("hello")
            peer_id = hello["peer_id"]
            if code:
                if len(code) != 6 or not code.isascii() or not code.isdigit():
                    raise ValueError("Enter the six-digit code shown in the iPhone app")
                reply, _ = client.request("pair", code=code)
                self.credentials.save(peer_id, reply["token"])
            client.token = self.credentials.get(peer_id)
            if not client.token:
                raise PairingRequired("Enter the pairing code shown on your iPhone")
            status, _ = client.request("status")
            self.client = client; self.serial = serial
            emit("connected", peer_id=peer_id, message="iPhone paired over USB", available_memory_bytes=status.get("available_memory_bytes", 0))
        except Exception:
            client.close(); raise

    def load(self, path, phone_layers=None, context=512):
        if not self.client:
            raise ValueError("Connect to your iPhone first")
        emit("progress", message="Checking your local model…", fraction=-1)
        checkpoint = inspect_model(path)
        # Validate architecture against the same runtime used for actual execution.
        from .runtime import checked_args
        checked_args(checkpoint.config)
        status, _ = self.client.request("status")
        self.context = int(context)
        end = choose_layers(checkpoint, status.get("available_memory_bytes", 0), phone_layers)
        physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if fp16_size(checkpoint.tensors.values()) > physical * .55:
            raise ValueError("This preview's conservative Mac memory limit is exceeded. Choose a smaller model.")
        serial = self.serial
        # Preparation can take minutes. Reconnect afterwards rather than leaving an idle socket open.
        self.disconnect()
        emit("progress", message="Preparing the model for your Mac and iPhone…", fraction=-1)
        directory = prepare(checkpoint, self.support / "Models", end, self.context, self.cancelled.is_set)
        if self.cancelled.is_set(): raise InterruptedError("Preparation cancelled")
        self.connect(serial)
        def progress(done, total):
            emit("progress", message="Transferring and verifying iPhone layers…", fraction=done / total)
        self.model_id = transfer(self.client, directory, progress, self.cancelled.is_set)
        emit("progress", message="Loading both devices…", fraction=-1)
        self.client.request("load", model_id=self.model_id)
        import mlx.core as mx
        from .runtime import MacCoordinator
        from transformers import AutoTokenizer
        remote = RemoteStage(self.client, checkpoint.config, 0, end)
        self.coordinator = MacCoordinator(checkpoint.config, directory / "mac.safetensors", 0, end, remote, max_context=self.context)
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint.root), local_files_only=True, trust_remote_code=False)
        self.model_path = str(checkpoint.root)
        emit("model_ready", name=checkpoint.root.name, phone_layers=end, context=self.context,
             mac_weight_bytes=self.coordinator.weight_bytes,
             iphone_weight_bytes=remote.weight_bytes)

    def generate(self, messages, max_tokens=128):
        if not self.coordinator or not self.tokenizer:
            raise ValueError("Load a model before starting a conversation")
        if not isinstance(messages, list) or not messages or len(messages) > 100:
            raise ValueError("Invalid conversation")
        for message in messages:
            if set(message) != {"role", "content"} or message["role"] not in ("user", "assistant", "system") or not isinstance(message["content"], str) or len(message["content"]) > 32768:
                raise ValueError("Invalid conversation message")
        maximum = max(1, min(int(max_tokens), 256))
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
        else:
            prompt = "\n".join(("Question: " if m["role"] == "user" else "Answer: ") + m["content"] for m in messages) + "\nAnswer:"
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("The prompt is empty")
        if len(ids) + maximum > self.context:
            raise ValueError(f"Conversation is too long for the {self.context}-token preview limit. Start a new chat or shorten your prompt.")
        import mlx.core as mx
        self.coordinator.reset()
        started = time.perf_counter(); generated = []; text = ""
        # Bound prefill activations and working memory; do not send a whole long prompt at once.
        for offset in range(0, len(ids), 32):
            if self.cancelled.is_set(): break
            logits = self.coordinator.forward(mx.array([ids[offset:offset+32]], mx.int32))
        if not self.cancelled.is_set():
            for _ in range(maximum):
                if self.cancelled.is_set(): break
                token = int(mx.argmax(logits[0, -1]).item())
                eos = self.tokenizer.eos_token_id
                if token in (eos if isinstance(eos, list) else [eos]): break
                generated.append(token)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                emit("text", text=text)
                if len(generated) < maximum:
                    logits = self.coordinator.forward(mx.array([[token]], mx.int32))
        emit("generation_done", text=text, tokens=len(generated), seconds=time.perf_counter()-started, cancelled=self.cancelled.is_set())
        self.coordinator.reset()


class RemoteStage:
    def __init__(self, client, config, start, end):
        self.client = client; self.start = start; self.end = end
        self.hidden_size = config["hidden_size"]
        status, _ = client.request("status")
        self.weight_bytes = status["weight_bytes"]

    def reset(self): self.client.request("reset")

    def forward(self, hidden, position):
        import mlx.core as mx
        import numpy as np
        mx.eval(hidden)
        reply, body = self.client.request("forward", np.array(hidden).astype("<f2").tobytes(), tokens=hidden.shape[1], position=position)
        if reply.get("shape") != list(hidden.shape) or reply.get("dtype") != "float16" or reply.get("position") != position + hidden.shape[1] or len(body) != hidden.size * 2:
            raise ValueError("iPhone returned an incompatible result")
        return mx.array(np.frombuffer(body, dtype="<f2").copy().reshape(hidden.shape))


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    engine = Engine(os.environ.get("MLX_PEER_SUPPORT"))
    commands = queue.Queue(maxsize=8)
    finished = threading.Event()
    def read_commands():
        for line in sys.stdin:
            try:
                if len(line) > 131072: raise ValueError("Command too large")
                command = json.loads(line)
                if command.get("command") == "cancel": engine.cancelled.set()
                else: commands.put_nowait(command)
            except Exception as error: emit("error", message=str(error))
        engine.cancelled.set(); finished.set()
    threading.Thread(target=read_commands, daemon=True).start()
    emit("ready", version="0.2.0-alpha.2")
    try:
        while not finished.is_set():
            try: command = commands.get(timeout=5)
            except queue.Empty:
                try:
                    engine.discover()
                    if engine.client: engine.client.request("status")
                except Exception:
                    if engine.client:
                        engine.disconnect(); emit("disconnected", message="iPhone disconnected. Open its app and reconnect.")
                continue
            engine.cancelled.clear()
            try:
                action = command.get("command")
                if action == "devices": engine.discover()
                elif action == "connect": engine.connect(command["serial"], command.get("code", ""))
                elif action == "load": engine.load(command["path"], command.get("phone_layers"), command.get("context", 512))
                elif action == "generate": engine.generate(command["messages"], command.get("max_tokens", 128))
                elif action == "disconnect": engine.disconnect(); emit("disconnected", message="Disconnected")
                else: raise ValueError("Unknown command")
            except PairingRequired as error: emit("pairing_required", message=str(error))
            except InterruptedError as error: engine.disconnect(); emit("cancelled", message=str(error))
            except Exception as error:
                if action == "generate" and isinstance(error, ValueError) and engine.coordinator and not engine.coordinator.failed:
                    emit("notice", message=str(error))
                else:
                    engine.disconnect()
                    emit("error", message=str(error) or type(error).__name__)
            finally: emit("done")
    finally: engine.disconnect()


if __name__ == "__main__": main()
