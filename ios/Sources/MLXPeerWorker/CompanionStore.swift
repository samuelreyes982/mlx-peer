import Foundation
import CryptoKit

/// Bounded, resumable imports. Only a complete, hash-verified bundle can become a model.
public final class CompanionStore {
    public static let files: Set<String> = ["config.json", "request.json", "weights.safetensors"]
    public static let maxWeights = 1_073_741_824
    public let root: URL
    private let fm = FileManager.default
    public init(root: URL) throws {
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        try fm.createDirectory(at: self.root, withIntermediateDirectories: true)
        var url = self.root
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }
    public static func validID(_ id: String) -> Bool {
        id.count == 64 && id.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    public func directory(_ id: String) throws -> URL {
        guard Self.validID(id) else { throw WorkerError.invalid("Invalid model identifier") }
        return try FixtureRunner.member(id, root: root)
    }
    private func member(_ id: String, _ name: String) throws -> URL {
        guard Self.files.contains(name) else { throw WorkerError.invalid("Unsupported model file") }
        return try FixtureRunner.member(name, root: directory(id))
    }
    public func begin(id: String, manifest: [String: Any]) throws -> [String: Any] {
        guard Self.validID(id), Set(manifest.keys) == Self.files else { throw WorkerError.invalid("Incomplete model manifest") }
        for (name, raw) in manifest {
            guard let item = raw as? [String: Any], let size = item["size"] as? Int,
                  size > 0, size <= (name == "weights.safetensors" ? Self.maxWeights : 65536),
                  let hash = item["sha256"] as? String, Self.validID(hash) else {
                throw WorkerError.invalid("Invalid model file size or checksum")
            }
        }
        let bytes = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        guard id == hash else { throw WorkerError.invalid("Model identifier does not match its manifest") }
        let directory = try self.directory(id)
        if !fm.fileExists(atPath: directory.path) {
            let needed = manifest.values.reduce(0) { $0 + (($1 as? [String: Any])?["size"] as? Int ?? 0) }
            let free = (try fm.attributesOfFileSystem(forPath: root.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
            guard free > Int64(needed) + 256 * 1024 * 1024 else { throw WorkerError.invalid("Not enough free storage on this iPhone") }
            try fm.createDirectory(at: directory, withIntermediateDirectories: false)
        }
        let manifestURL = try FixtureRunner.member("manifest.json", root: directory)
        if fm.fileExists(atPath: manifestURL.path) {
            guard try Data(contentsOf: manifestURL) == bytes else { throw WorkerError.invalid("Model identity conflicts with an existing transfer") }
        } else { try bytes.write(to: manifestURL, options: .atomic) }
        var offsets: [String: Int] = [:]
        for name in Self.files {
            let file = try member(id, name)
            offsets[name] = ((try? fm.attributesOfItem(atPath: file.path)[.size]) as? NSNumber)?.intValue ?? 0
            let limit = (manifest[name] as! [String: Any])["size"] as! Int
            guard offsets[name]! <= limit else { throw WorkerError.invalid("Stored transfer exceeds expected file size; remove it and retry") }
        }
        return ["offsets": offsets]
    }
    private func manifest(_ id: String) throws -> [String: Any] {
        let url = try FixtureRunner.member("manifest.json", root: directory(id))
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any], Set(value.keys) == Self.files else { throw WorkerError.invalid("Stored manifest is damaged; remove saved models and retry") }
        for raw in value.values {
            guard let entry = raw as? [String: Any], let size = entry["size"] as? Int, size > 0, size <= Self.maxWeights, let hash = entry["sha256"] as? String, Self.validID(hash) else { throw WorkerError.invalid("Stored manifest is damaged; remove saved models and retry") }
        }
        return value
    }
    public func append(id: String, name: String, offset: Int, data: Data) throws -> Int {
        guard !data.isEmpty, data.count <= 4 * 1024 * 1024, offset >= 0 else { throw WorkerError.invalid("Invalid transfer chunk") }
        let url = try member(id, name)
        let entry = try manifest(id)[name] as! [String: Any]
        let expected = entry["size"] as! Int
        guard offset <= expected, data.count <= expected - offset else { throw WorkerError.invalid("Transfer exceeds declared size") }
        let ready = try FixtureRunner.member("ready", root: directory(id))
        guard !fm.fileExists(atPath: ready.path) else { throw WorkerError.invalid("A completed model cannot be modified") }
        if !fm.fileExists(atPath: url.path) { guard offset == 0 else { throw WorkerError.invalid("Missing transfer prefix") }; fm.createFile(atPath: url.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: url); defer { try? handle.close() }
        guard try handle.seekToEnd() == UInt64(offset) else { throw WorkerError.invalid("Transfer offset mismatch") }
        try handle.write(contentsOf: data)
        try handle.synchronize()
        return offset + data.count
    }
    public func commit(_ id: String) throws {
        let entries = try manifest(id)
        for name in Self.files {
            let entry = entries[name] as! [String: Any]
            let url = try member(id, name)
            guard let size = try fm.attributesOfItem(atPath: url.path)[.size] as? NSNumber,
                  size.intValue == entry["size"] as? Int,
                  try WireServer.digest(url) == entry["sha256"] as? String else {
                throw WorkerError.invalid("Model checksum failed. Remove this model and transfer it again.")
            }
        }
        try Data(id.utf8).write(to: FixtureRunner.member("ready", root: directory(id)), options: .atomic)
    }
    public func readyDirectory(_ id: String) throws -> URL {
        let directory = try self.directory(id)
        guard fm.fileExists(atPath: try FixtureRunner.member("ready", root: directory).path) else {
            throw WorkerError.invalid("Finish transferring the model first")
        }
        return directory
    }
    public func remove(_ id: String) throws { let url = try directory(id); if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) } }
    public func removeAll() throws {
        for item in try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) where Self.validID(item.lastPathComponent) {
            try remove(item.lastPathComponent)
        }
    }
}
