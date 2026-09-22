import Foundation
import Darwin
import CryptoKit
import MLX
#if os(iOS)
import os
#endif

struct WireRequest: Decodable {
    let protocol_version: Int
    let request_id: Int
    let token: String
    let operation: String
    let payload_bytes: Int
    let position: Int?
    let tokens: Int?
    let bytes: Int?

    func validate(token expected: String) throws {
        let lhs = Array(token.utf8), rhs = Array(expected.utf8)
        guard lhs.count == 64, rhs.count == 64 else { throw WorkerError.invalid("Invalid token") }
        var difference: UInt8 = 0
        for i in lhs.indices { difference |= lhs[i] ^ rhs[i] }
        guard difference == 0, protocol_version == 1, request_id > 0,
              (0...WireServer.maxPayload).contains(payload_bytes),
              ["ping", "echo", "upload", "download", "status", "reset", "forward", "stop"].contains(operation) else {
            throw WorkerError.invalid("Invalid wire request")
        }
        if !["echo", "upload", "forward"].contains(operation), payload_bytes != 0 {
            throw WorkerError.invalid("Unexpected request payload")
        }
        if operation == "download", !(0...WireServer.maxPayload).contains(bytes ?? -1) {
            throw WorkerError.invalid("Invalid download length")
        }
    }
}

/// A foreground developer probe. Loopback only; usbmux provides the physical USB path.
/// One client owns the stage at a time. Disconnect/error always resets cached state.
public enum WireServer {
    static let maxPayload = 8 * 1024 * 1024
    static let maxHeader = 16 * 1024

    static func receive(_ fd: Int32, count: Int) throws -> Data {
        guard (0...maxPayload).contains(count) else { throw WorkerError.invalid("Invalid receive size") }
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { buffer in
            var offset = 0
            while offset < count {
                let n = Darwin.recv(fd, buffer.baseAddress!.advanced(by: offset), count - offset, 0)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw WorkerError.invalid("Connection ended or timed out") }
                offset += n
            }
        }
        return data
    }

    static func send(_ fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < data.count {
                let n = Darwin.send(fd, buffer.baseAddress!.advanced(by: offset), data.count - offset, 0)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw WorkerError.invalid("Send failed or timed out") }
                offset += n
            }
        }
    }

    static func respond(_ fd: Int32, id: Int, fields: [String: Any] = [:], body: Data = Data()) throws {
        var header = fields
        header["request_id"] = id; header["protocol_version"] = 1
        header["payload_bytes"] = body.count
        if header["ok"] == nil { header["ok"] = true }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        guard json.count <= maxHeader else { throw WorkerError.invalid("Reply header too large") }
        var length = UInt32(json.count).bigEndian
        var frame = withUnsafeBytes(of: &length) { Data($0) }
        frame.append(json); frame.append(body)
        try send(fd, data: frame)
    }

    static func digest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        // Foundation's autoreleased read buffers must not accumulate for the
        // lifetime of the foreground server (multi-gigabyte checkpoint files).
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let data = try file.read(upToCount: 1024 * 1024), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func run(directory: URL, port: UInt16 = 49172,
                           ready: @escaping (String) -> Void) throws -> String {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let tokenURL = try FixtureRunner.member("wire-token.txt", root: root)
        let tokenAttributes = try FileManager.default.attributesOfItem(atPath: tokenURL.path)
        guard (tokenAttributes[.size] as? NSNumber)?.intValue == 64 else {
            throw WorkerError.invalid("Missing 64-byte wire token")
        }
        let token = try String(contentsOf: tokenURL, encoding: .utf8)
        guard token.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw WorkerError.invalid("Invalid token encoding")
        }
        var loadMemory: [String: Int] = [:]
        #if os(iOS)
        loadMemory["before_stage_load_available_bytes"] = Int(os_proc_available_memory())
        #endif
        let (_, stage) = try FixtureRunner.loadStage(directory: root)
        #if os(iOS)
        loadMemory["after_stage_load_available_bytes"] = Int(os_proc_available_memory())
        #endif
        let weightHash = try digest(FixtureRunner.member("weights.safetensors", root: root))
        let configHash = try digest(FixtureRunner.member("config.json", root: root))
        guard stage.dtype == .float16 || stage.dtype == .bfloat16 else { throw WorkerError.invalid("Wire stage requires FP16 or BF16") }
        let listener = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard listener >= 0 else { throw WorkerError.invalid("Could not create listener") }
        defer { Darwin.close(listener) }
        var yes: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, Darwin.listen(listener, 1) == 0 else {
            throw WorkerError.invalid("Could not bind loopback probe port")
        }
        let deadline = ProcessInfo.processInfo.systemUptime + 600
        let download = Data(repeating: 0x5a, count: maxPayload)
        ready("USB probe ready on port \(port). Keep this app in the foreground.")
        var operations = 0
        while ProcessInfo.processInfo.systemUptime < deadline, operations < 4096 {
            var pollFD = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
            guard Darwin.poll(&pollFD, 1, 1000) > 0 else { continue }
            let fd = Darwin.accept(listener, nil, nil)
            guard fd >= 0 else { continue }
            defer { Darwin.close(fd) }
            var timeout = timeval(tv_sec: 120, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            stage.reset()
            var sequence = 0
            do {
                while ProcessInfo.processInfo.systemUptime < deadline, operations < 4096 {
                    let prefix = try receive(fd, count: 4)
                    let size = prefix.reduce(0) { ($0 << 8) | Int($1) }
                    guard (1...maxHeader).contains(size) else { throw WorkerError.invalid("Invalid header length") }
                    let request = try JSONDecoder().decode(WireRequest.self, from: receive(fd, count: size))
                    try request.validate(token: token)
                    guard request.request_id == sequence + 1 else { throw WorkerError.invalid("Sequence mismatch") }
                    sequence = request.request_id; operations += 1
                    let body = try receive(fd, count: request.payload_bytes)
                    var fields: [String: Any] = [:]
                    var output = Data()
                    do {
                        switch request.operation {
                        case "ping": break
                        case "echo": output = body
                        case "upload":
                            fields["sha256"] = SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()
                        case "download": output = download.prefix(request.bytes!)
                        case "status":
                            fields = ["weight_bytes": stage.weightBytes, "weights_sha256": weightHash,
                                "config_sha256": configHash, "start": stage.layerRange.lowerBound,
                                "end": stage.layerRange.upperBound, "position": stage.position,
                                "max_context": stage.maxContext, "hidden_size": stage.hiddenSize,
                                "dtype": String(describing: stage.dtype), "mlx_active_bytes": Memory.activeMemory,
                                "system": ProcessInfo.processInfo.operatingSystemVersionString,
                                "load_memory": loadMemory]
                        case "reset": stage.reset()
                        case "forward":
                            guard let count = request.tokens, let position = request.position,
                                  count > 0, count <= stage.maxContext,
                                  position >= 0, position <= stage.maxContext - count,
                                  body.count == count * stage.hiddenSize * 2 else {
                                throw WorkerError.invalid("Invalid activation shape or length")
                            }
                            let input = MLXArray(body, [1, count, stage.hiddenSize], dtype: stage.dtype)
                            eval(input)
                            let start = ProcessInfo.processInfo.systemUptime
                            let result = try stage.forward(input, position: position)
                            fields["compute_ms"] = (ProcessInfo.processInfo.systemUptime - start) * 1000
                            output = result.asData(access: .copy).data
                            fields["position"] = stage.position
                            fields["dtype"] = String(describing: stage.dtype)
                            fields["shape"] = result.shape
                            fields["mlx_active_bytes"] = Memory.activeMemory
                            fields["mlx_peak_bytes"] = Memory.peakMemory
                        case "stop":
                            try respond(fd, id: sequence)
                            return "USB probe finished after \(operations) requests."
                        default: throw WorkerError.invalid("Unsupported operation")
                        }
                        #if os(iOS)
                        fields["available_process_memory_bytes"] = Int(os_proc_available_memory())
                        #endif
                        try respond(fd, id: sequence, fields: fields, body: output)
                    } catch {
                        stage.reset()
                        try? respond(fd, id: sequence, fields: ["ok": false, "error": error.localizedDescription])
                        throw error
                    }
                }
            } catch {
                // This connection cannot be resumed after a partial request or failed stage.
                stage.reset()
            }
        }
        return "USB probe stopped at its time or request limit."
    }
}
