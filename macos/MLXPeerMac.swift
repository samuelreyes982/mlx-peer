import SwiftUI
import AppKit

struct Device: Identifiable { let id: String; let name: String }
struct Message: Identifiable { let id = UUID(); let role: String; var content: String }

@MainActor
final class DesktopModel: ObservableObject {
    @Published var devices: [Device] = []
    @Published var selectedDevice = ""
    @Published var pairingCode = ""
    @Published var needsPairing = false
    @Published var connected = false
    @Published var loaded = false
    @Published var busy = true
    @Published var generating = false
    @Published var status = "Starting local engine…"
    @Published var error = ""
    @Published var progress: Double = -1
    @Published var modelURL: URL?
    @Published var modelName = "No model selected"
    @Published var allocation = ""
    @Published var messages: [Message] = []
    @Published var prompt = ""
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var diagnostics: FileHandle?
    private var pending = Data()

    init() {
        if let path = UserDefaults.standard.string(forKey: "lastModelFolder") {
            modelURL = URL(fileURLWithPath: path, isDirectory: true); modelName = modelURL!.lastPathComponent
        }
        start()
    }
    func start() {
        let executable = Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/MLX Peer Engine.app/Contents/MacOS/mlx-peer-engine")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            error = "The bundled engine is missing. Download a complete MLX Peer app."; busy = false; return
        }
        let task = Process(); task.executableURL = executable
        task.currentDirectoryURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        var environment = ProcessInfo.processInfo.environment
        for key in ["PYTHONPATH", "PYTHONHOME", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"] { environment.removeValue(forKey: key) }
        environment["PYTHONUNBUFFERED"] = "1"
        task.environment = environment
        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        task.standardInput = stdin; task.standardOutput = stdout; task.standardError = stderr
        input = stdin.fileHandleForWriting; output = stdout.fileHandleForReading; diagnostics = stderr.fileHandleForReading
        output?.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async { self.receive(data) }
        }
        // Drain diagnostics without storing prompts, credentials, or private model paths on disk.
        diagnostics?.readabilityHandler = { handle in if handle.availableData.isEmpty { handle.readabilityHandler = nil } }
        task.terminationHandler = { task in DispatchQueue.main.async {
            self.connected = false; self.loaded = false; self.busy = false; self.generating = false
            self.status = "Engine stopped"
            if task.terminationStatus != 0 { self.error = "The local engine stopped unexpectedly. Quit and reopen MLX Peer." }
        } }
        do { try task.run(); process = task }
        catch { self.error = "Could not start the local engine: \(error.localizedDescription)"; busy = false }
    }
    func shutdown() { try? input?.close(); if process?.isRunning == true { process?.terminate() } }
    private func send(_ command: [String: Any]) {
        guard let input, let data = try? JSONSerialization.data(withJSONObject: command) else { return }
        do { try input.write(contentsOf: data + Data([10])) }
        catch { self.error = "The local engine is no longer available. Reopen the app."; busy = false }
    }
    private func receive(_ data: Data) {
        pending.append(data)
        while let newline = pending.firstIndex(of: 10) {
            let line = pending[..<newline]; pending.removeSubrange(...newline)
            guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any], let kind = event["event"] as? String else { continue }
            switch kind {
            case "ready": busy = false; status = "Open MLX Peer on your iPhone, then connect its USB cable."; send(["command": "devices"])
            case "devices":
                devices = (event["devices"] as? [[String: String]] ?? []).compactMap { d in guard let id = d["id"] else { return nil }; return Device(id: id, name: d["name"] ?? "iPhone") }
                if !devices.contains(where: { $0.id == selectedDevice }) { selectedDevice = devices.first?.id ?? "" }
            case "connected": connected = true; needsPairing = false; pairingCode = ""; status = "iPhone paired over USB"
            case "pairing_required": connected = false; loaded = false; needsPairing = true; status = event["message"] as? String ?? "Enter your iPhone's code"
            case "progress": progress = event["fraction"] as? Double ?? -1; status = event["message"] as? String ?? "Working…"
            case "model_ready":
                loaded = true; modelName = event["name"] as? String ?? "Local model"; messages = []
                let mac = ByteCountFormatter.string(fromByteCount: (event["mac_weight_bytes"] as? NSNumber)?.int64Value ?? 0, countStyle: .memory)
                let phone = ByteCountFormatter.string(fromByteCount: (event["iphone_weight_bytes"] as? NSNumber)?.int64Value ?? 0, countStyle: .memory)
                allocation = "\(mac) on Mac · \(phone) on iPhone\n\(event["phone_layers"] as? Int ?? 0) phone layers · \(event["context"] as? Int ?? 512)-token context"
                status = "Both devices are ready. Start a conversation."; progress = -1
            case "text":
                if messages.last?.role == "assistant" { messages[messages.count - 1].content = event["text"] as? String ?? "" }
            case "generation_done":
                generating = false
                status = (event["cancelled"] as? Bool == true) ? "Generation stopped" : "Generated \(event["tokens"] as? Int ?? 0) tokens across your Mac and iPhone"
                if messages.last?.content.isEmpty == true { messages.removeLast() }
            case "notice": error = event["message"] as? String ?? "Try a shorter prompt"; generating = false
            case "error", "cancelled", "disconnected":
                connected = false; loaded = false; generating = false; busy = false; progress = -1
                status = event["message"] as? String ?? "Reconnect your iPhone"
                if kind == "error" { error = status }
            case "done": busy = false; if !loaded { generating = false }
            default: break
            }
        }
        if pending.count > 2 * 1024 * 1024 { pending.removeAll(); error = "The engine returned an unreadable response." }
    }
    func connect() {
        guard !selectedDevice.isEmpty else { return }
        busy = true; error = ""; loaded = false; status = "Connecting to your iPhone…"
        send(["command": "connect", "serial": selectedDevice, "code": pairingCode.trimmingCharacters(in: .whitespacesAndNewlines)])
    }
    func disconnect() { busy = true; send(["command": "disconnect"]) }
    func selectModel() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
        panel.message = "Choose a local Qwen2 / Qwen2.5 folder with config.json, tokenizer.json, and safetensors weights. An Instruct model is recommended for chat."
        panel.prompt = "Choose model"
        if panel.runModal() == .OK, let url = panel.url {
            modelURL = url; modelName = url.lastPathComponent; loaded = false; allocation = ""; messages = []
            UserDefaults.standard.set(url.path, forKey: "lastModelFolder")
        }
    }
    func load() {
        guard let modelURL else { return }
        busy = true; loaded = false; error = ""; progress = -1
        send(["command": "load", "path": modelURL.path])
    }
    func generate() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard loaded, !busy, !text.isEmpty else { return }
        error = ""; messages.append(Message(role: "user", content: text)); prompt = ""
        let conversation = messages.filter { !$0.content.isEmpty }.map { ["role": $0.role, "content": $0.content] }
        messages.append(Message(role: "assistant", content: "")); busy = true; generating = true
        status = "Thinking across both devices…"
        send(["command": "generate", "messages": conversation, "max_tokens": 128])
    }
    func cancel() { status = "Stopping…"; send(["command": "cancel"]) }
}

@main
struct MLXPeerMacApp: App {
    @StateObject private var model = DesktopModel()
    var body: some Scene {
        WindowGroup("MLX Peer") {
            DesktopView(model: model)
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in model.shutdown() }
        }.defaultSize(width: 1040, height: 740)
        .commands { CommandGroup(replacing: .newItem) { Button("New chat") { model.messages = [] }.keyboardShortcut("n").disabled(model.busy) } }
    }
}

struct DesktopView: View {
    @ObservedObject var model: DesktopModel
    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 24) {
                HStack(spacing: 10) {
                    Image(systemName: "point.3.connected.trianglepath.dotted").font(.system(size: 28)).foregroundStyle(.blue)
                    VStack(alignment: .leading, spacing: 2) { Text("MLX Peer").font(.title2.bold()); Text("LOCAL AI, TOGETHER").font(.system(size: 9, weight: .semibold)).tracking(1.3).foregroundStyle(.secondary) }
                }.padding(.top, 12)
                VStack(alignment: .leading, spacing: 12) {
                    Label("1. Connect your iPhone", systemImage: "cable.connector").font(.headline)
                    Text("Open the iPhone companion and keep it unlocked. Use a data-capable USB cable and trust this Mac when iOS asks.").font(.callout).foregroundStyle(.secondary)
                    if model.devices.isEmpty { Text("Waiting for a USB iPhone…").font(.callout).foregroundStyle(.secondary) }
                    else {
                        Picker("Device", selection: $model.selectedDevice) { ForEach(Array(model.devices.enumerated()), id: \.element.id) { index, device in Text(model.devices.count > 1 ? "iPhone \(index + 1)" : device.name).tag(device.id) } }.labelsHidden().disabled(model.busy || model.connected)
                    }
                    if !model.connected {
                        TextField("6-digit iPhone code (first pairing)", text: $model.pairingCode).textFieldStyle(.roundedBorder).disabled(model.busy)
                            .onSubmit { if !model.busy { model.connect() } }
                        Text("Already paired? Leave the code blank.").font(.caption).foregroundStyle(.secondary)
                    }
                    Button(model.connected ? "Disconnect iPhone" : "Connect iPhone") { model.connected ? model.disconnect() : model.connect() }
                        .buttonStyle(.borderedProminent).disabled(model.busy || model.selectedDevice.isEmpty)
                }
                Divider()
                VStack(alignment: .leading, spacing: 12) {
                    Label("2. Choose a local model", systemImage: "folder").font(.headline)
                    Text(model.modelName).font(.callout.weight(.medium)).lineLimit(2)
                    Button("Choose model folder…", action: model.selectModel).disabled(model.busy)
                    Text("Qwen2 / Qwen2.5 safetensors with tokenizer files. Choose an Instruct model for chat. FP16, BF16, or FP32 input; prepares FP16 weights.").font(.caption).foregroundStyle(.secondary)
                    Button("Load on both devices", action: model.load).disabled(!model.connected || model.modelURL == nil || model.busy)
                    if !model.allocation.isEmpty { Text(model.allocation).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                }
                Spacer(minLength: 4)
                VStack(alignment: .leading, spacing: 8) {
                    Label("Your hardware. Your model.", systemImage: "lock.shield").font(.callout.weight(.medium))
                    Text("No cloud inference. Models and prompts stay on your devices.").font(.caption).foregroundStyle(.secondary)
                    Text("Developer preview 0.2 · Apple Silicon").font(.caption2).foregroundStyle(.tertiary)
                }
            }.padding(24).frame(width: 300).frame(maxHeight: .infinity).background(Color(nsColor: .controlBackgroundColor))
            Divider()
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Intelligence with what you own.").font(.title2.weight(.semibold))
                        Label(model.connected ? "iPhone connected over USB" : "Mac + iPhone", systemImage: model.connected ? "checkmark.circle.fill" : "laptopcomputer.and.iphone").font(.callout).foregroundStyle(model.connected ? .green : .secondary)
                    }
                    Spacer()
                    Button("New chat") { model.messages = []; model.error = "" }.disabled(model.busy || model.messages.isEmpty)
                }.padding(24)
                Divider()
                ScrollViewReader { proxy in
                    ScrollView {
                        if model.messages.isEmpty {
                            VStack(spacing: 18) {
                                Image(systemName: "laptopcomputer.and.iphone").font(.system(size: 66, weight: .light)).foregroundStyle(.blue)
                                Text("One model. Two devices.").font(.title.bold())
                                Text("Connect your iPhone and select a local model. MLX Peer prepares each device's share, verifies the transfer, and brings them together for your conversation.")
                                    .font(.body).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 430)
                                Text("Start small. This preview supports short conversations and does not automatically download models.").font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 400)
                            }.frame(maxWidth: .infinity).padding(.vertical, 65).padding(.horizontal, 24)
                        } else {
                            LazyVStack(alignment: .leading, spacing: 20) {
                                ForEach(model.messages) { message in
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text(message.role == "user" ? "YOU" : "MLX PEER").font(.system(size: 10, weight: .bold)).tracking(1).foregroundStyle(.secondary)
                                        Text(message.content.isEmpty ? "Thinking…" : message.content).font(.body).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                                    }.padding(18).background(message.role == "user" ? Color.blue.opacity(0.07) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14)).id(message.id)
                                }
                                Color.clear.frame(height: 1).id("bottom")
                            }.padding(24)
                        }
                    }.onChange(of: model.messages.last?.content) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                }
                VStack(alignment: .leading, spacing: 10) {
                    if !model.error.isEmpty { Label(model.error, systemImage: "exclamationmark.circle").foregroundStyle(.red).font(.callout).textSelection(.enabled) }
                    HStack(spacing: 10) {
                        if model.busy { ProgressView().controlSize(.small) }
                        Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(3)
                        Spacer()
                        if model.busy { Button("Stop", action: model.cancel).controlSize(.small) }
                    }
                    if model.busy && model.progress >= 0 { ProgressView(value: model.progress) }
                    HStack(alignment: .bottom, spacing: 10) {
                        TextField("Ask your local model…", text: $model.prompt, axis: .vertical).lineLimit(1...5).textFieldStyle(.plain).padding(14).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12)).disabled(!model.loaded || model.busy)
                            .onSubmit { model.generate() }
                        Button(action: model.generate) { Image(systemName: "arrow.up").font(.headline).padding(6) }.buttonStyle(.borderedProminent).disabled(!model.loaded || model.busy || model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty).accessibilityLabel("Send message")
                    }
                    Text("512-token context · Generated answers can be incorrect.").font(.caption2).foregroundStyle(.tertiary)
                }.padding(20)
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }.frame(minWidth: 900, minHeight: 660)
    }
}
