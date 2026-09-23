import Foundation
import CryptoKit
import Darwin
import MLX
#if os(iOS)
import os
#endif

/// Foreground, USB-only companion protocol. No model is required to start pairing.
public final class CompanionServer: @unchecked Sendable {
    public static let port: UInt16 = 49173
    private let lock = NSLock()
    private var stopping = false
    private var socket: Int32 = -1
    private var token: String
    private var code: String
    private var expires: Date
    private var attempts = 0
    public let peerID: String
    private let credentialURL: URL
    private let store: CompanionStore
    public var pairingCode: String { lock.lock(); defer { lock.unlock() }; return code }
    public var isPaired: Bool { lock.lock(); defer { lock.unlock() }; return !token.isEmpty }
    private var shouldStop: Bool { lock.lock(); defer { lock.unlock() }; return stopping }

    public init(root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        credentialURL = root.appendingPathComponent("pairing.json")
        let saved = (try? Data(contentsOf: credentialURL)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: String] }
        peerID = saved?["peer_id"] ?? UUID().uuidString.lowercased()
        token = saved?["token"] ?? ""
        code = String(format: "%06d", Int.random(in: 0...999999))
        expires = Date().addingTimeInterval(300)
        store = try CompanionStore(root: root.appendingPathComponent("Models", isDirectory: true))
        try persist()
    }
    private func persist() throws {
        let data = try JSONSerialization.data(withJSONObject: ["peer_id": peerID, "token": token])
        #if os(iOS)
        try data.write(to: credentialURL, options: [.atomic, .completeFileProtection])
        #else
        try data.write(to: credentialURL, options: .atomic)
        #endif
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: credentialURL.path)
        var url = credentialURL; var values = URLResourceValues(); values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }
    /// Call only while the server is stopped; replacing a pairing invalidates the old Mac token.
    public func forgetPairing() throws {
        lock.lock(); defer { lock.unlock() }
        token = ""; code = String(format: "%06d", Int.random(in: 0...999999)); attempts = 0
        expires = Date().addingTimeInterval(300); try persist()
    }
    public func removeModels() throws { try store.removeAll() }
    public func stop() {
        lock.lock(); stopping = true
        if socket >= 0 { Darwin.shutdown(socket, SHUT_RDWR) }
        lock.unlock()
    }
    private func setClient(_ fd: Int32) { lock.lock(); socket = fd; lock.unlock() }
    private func authenticate(_ value: String?) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !token.isEmpty, let value, value.utf8.count == token.utf8.count else { return false }
        return zip(value.utf8, token.utf8).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) } == 0
    }
    private func pair(_ supplied: String?) throws -> String {
        lock.lock(); defer { lock.unlock() }
        guard token.isEmpty, Date() < expires, attempts < 5 else {
            throw WorkerError.invalid("Pairing is closed. On iPhone, stop sharing and choose Pair a new Mac.")
        }
        attempts += 1
        guard supplied == code else { throw WorkerError.invalid("The pairing code is incorrect") }
        token = (0..<32).map { _ in String(format: "%02x", UInt8.random(in: .min ... .max)) }.joined()
        try persist(); return token
    }
    private func respond(_ fd: Int32, id: Int, fields: [String: Any] = [:], body: Data = Data()) throws {
        var value = fields
        value["protocol_version"] = 2; value["request_id"] = id; value["payload_bytes"] = body.count
        if value["ok"] == nil { value["ok"] = true }
        let json = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        guard json.count <= 16384 else { throw WorkerError.invalid("Reply too large") }
        var size = UInt32(json.count).bigEndian
        var frame = withUnsafeBytes(of: &size) { Data($0) }; frame.append(json); frame.append(body)
        try WireServer.send(fd, data: frame)
    }
    public func run(update: @escaping @Sendable (String) -> Void) throws {
        let listener = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard listener >= 0 else { throw WorkerError.invalid("Unable to start USB sharing") }
        defer { Darwin.close(listener) }
        var yes: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in(); address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET); address.sin_port = Self.port.bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        guard bound == 0, Darwin.listen(listener, 1) == 0 else { throw WorkerError.invalid("USB sharing is already running") }
        update("Ready for your Mac. Keep this app open.")
        while !shouldStop {
            var event = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
            guard Darwin.poll(&event, 1, 250) > 0 else { continue }
            let fd = Darwin.accept(listener, nil, nil); guard fd >= 0 else { continue }
            setClient(fd)
            var timeout = timeval(tv_sec: 30, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            do { try serve(fd, update: update) }
            catch { if !shouldStop { update("Connection ended. Reconnect from your Mac.") } }
            setClient(-1); Darwin.close(fd); Memory.clearCache()
        }
        update("Sharing stopped. Saved models stay on this iPhone.")
    }
    private func serve(_ fd: Int32, update: @escaping @Sendable (String) -> Void) throws {
        var stage: (any ModelStage)?
        var loadedID: String?
        var sequence = 0
        while !shouldStop {
            let prefix = try WireServer.receive(fd, count: 4)
            let length = prefix.reduce(0) { ($0 << 8) | Int($1) }
            guard (1...16384).contains(length) else { throw WorkerError.invalid("Invalid request header") }
            let raw = try WireServer.receive(fd, count: length)
            guard let request = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
                  request["protocol_version"] as? Int == 2,
                  let id = request["request_id"] as? Int, id == sequence + 1,
                  let operation = request["operation"] as? String,
                  let size = request["payload_bytes"] as? Int, (0...8*1024*1024).contains(size) else {
                throw WorkerError.invalid("Unsupported or out-of-sequence request")
            }
            sequence = id
            guard operation == "hello" || operation == "pair" || authenticate(request["token"] as? String) else {
                try respond(fd, id: id, fields: ["ok": false, "error": "Pair with this iPhone first.", "error_code": "pairing_required"])
                return
            }
            let body = try WireServer.receive(fd, count: size)
            do {
                if !["chunk", "forward"].contains(operation), !body.isEmpty { throw WorkerError.invalid("Unexpected payload") }
                var fields: [String: Any] = [:]; var output = Data()
                switch operation {
                case "hello": fields = ["peer_id": peerID, "paired": isPaired, "app_version": "0.2.0-alpha.1"]
                case "pair": fields = ["token": try pair(request["code"] as? String), "peer_id": peerID]; update("Mac paired. Waiting for a model.")
                case "status":
                    fields = ["peer_id": peerID, "max_shard_bytes": CompanionStore.maxWeights, "max_context": 1024, "model_id": loadedID ?? "", "ready": stage != nil]
                    #if os(iOS)
                    fields["available_memory_bytes"] = Int(os_proc_available_memory())
                    #endif
                    if let stage { fields["weight_bytes"] = stage.weightBytes; fields["dtype"] = String(describing: stage.dtype); fields["hidden_size"] = stage.hiddenSize; fields["start"] = stage.layerRange.lowerBound; fields["end"] = stage.layerRange.upperBound }
                case "begin":
                    guard let model = request["model_id"] as? String, let manifest = request["files"] as? [String: Any] else { throw WorkerError.invalid("Missing model manifest") }
                    stage = nil; loadedID = nil; Memory.clearCache()
                    fields = try store.begin(id: model, manifest: manifest); update("Receiving a model from your Mac…")
                case "chunk":
                    guard let model = request["model_id"] as? String, let name = request["file"] as? String, let offset = request["offset"] as? Int else { throw WorkerError.invalid("Missing transfer position") }
                    fields["offset"] = try store.append(id: model, name: name, offset: offset, data: body)
                case "commit":
                    guard let model = request["model_id"] as? String else { throw WorkerError.invalid("Missing model") }
                    try store.commit(model); update("Model transfer verified.")
                case "load":
                    guard let model = request["model_id"] as? String else { throw WorkerError.invalid("Missing model") }
                    let directory = try store.readyDirectory(model)
                    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
                    guard config?["model_type"] as? String == "qwen2" else { throw WorkerError.invalid("This companion release supports dense Qwen2 / Qwen2.5 models") }
                    let limits = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("request.json"))) as? [String: Any]
                    guard let context = limits?["max_context"] as? Int, (64...1024).contains(context) else { throw WorkerError.invalid("Context must be between 64 and 1024 tokens") }
                    stage = nil; Memory.clearCache()
                    let bytes = (try FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent("weights.safetensors").path)[.size] as! NSNumber).intValue
                    #if os(iOS)
                    guard bytes + 512 * 1024 * 1024 < Int(os_proc_available_memory()) else { throw WorkerError.invalid("Not enough available memory. Choose fewer phone layers.") }
                    #endif
                    let (request, loaded) = try FixtureRunner.loadStage(directory: directory)
                    guard request.maxContext <= 1024, loaded.dtype == .float16 else { throw WorkerError.invalid("Only FP16 models with at most 1024 context tokens are supported") }
                    stage = loaded; loadedID = model
                    fields = ["weight_bytes": loaded.weightBytes, "dtype": String(describing: loaded.dtype), "hidden_size": loaded.hiddenSize, "start": loaded.layerRange.lowerBound, "end": loaded.layerRange.upperBound]
                    update("Connected • \(loaded.layerRange.count) model layers ready")
                case "reset": guard let stage else { throw WorkerError.invalid("Load a model first") }; stage.reset()
                case "forward":
                    guard let stage, let count = request["tokens"] as? Int, let position = request["position"] as? Int,
                          count > 0, count <= 64, position >= 0, position <= stage.maxContext - count,
                          body.count == count * stage.hiddenSize * 2 else { throw WorkerError.invalid("Invalid model input") }
                    guard ProcessInfo.processInfo.thermalState != .critical && ProcessInfo.processInfo.thermalState != .serious else { throw WorkerError.invalid("iPhone needs to cool down. Stop and try again later.") }
                    let input = MLXArray(body, [1, count, stage.hiddenSize], dtype: .float16)
                    let result = try stage.forward(input, position: position)
                    output = result.asData(access: .copy).data
                    fields = ["shape": result.shape, "dtype": "float16", "position": stage.position]
                    update("Computing with your Mac • \(stage.position) tokens")
                case "unload": stage = nil; loadedID = nil; Memory.clearCache(); update("Mac connected. Model unloaded.")
                case "remove":
                    guard let model = request["model_id"] as? String else { throw WorkerError.invalid("Missing model") }
                    if model == loadedID { stage = nil; loadedID = nil; Memory.clearCache() }
                    try store.remove(model)
                case "disconnect": try respond(fd, id: id); return
                default: throw WorkerError.invalid("Unsupported operation")
                }
                try respond(fd, id: id, fields: fields, body: output)
            } catch {
                stage?.reset()
                try respond(fd, id: id, fields: ["ok": false, "error": error.localizedDescription])
            }
        }
    }
}
