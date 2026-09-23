import XCTest
import CryptoKit
@testable import MLXPeerWorker

final class CompanionTests: XCTestCase {
    private func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    func testResumeChecksumAndImmutableCommit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try CompanionStore(root: root)
        let data = Data("model fixture".utf8)
        let manifest = Dictionary(uniqueKeysWithValues: CompanionStore.files.map { ($0, ["size": data.count, "sha256": digest(data)] as [String: Any]) })
        let id = digest(try JSONSerialization.data(withJSONObject: manifest, options: .sortedKeys))
        _ = try store.begin(id: id, manifest: manifest)
        XCTAssertThrowsError(try store.readyDirectory(id))
        XCTAssertThrowsError(try store.append(id: id, name: "../outside", offset: 0, data: data))
        XCTAssertThrowsError(try store.append(id: id, name: "config.json", offset: 1, data: data))
        XCTAssertEqual(try store.append(id: id, name: "config.json", offset: 0, data: data.prefix(3)), 3)
        let resumed = try store.begin(id: id, manifest: manifest)["offsets"] as? [String: Int]
        XCTAssertEqual(resumed?["config.json"], 3)
        XCTAssertThrowsError(try store.commit(id))
        _ = try store.append(id: id, name: "config.json", offset: 3, data: data.dropFirst(3))
        for name in CompanionStore.files where name != "config.json" { _ = try store.append(id: id, name: name, offset: 0, data: data) }
        try store.commit(id)
        XCTAssertEqual(try store.readyDirectory(id).lastPathComponent, id)
        XCTAssertThrowsError(try store.append(id: id, name: "config.json", offset: 0, data: data))
        try store.remove(id)
        XCTAssertFalse(FileManager.default.fileExists(atPath: try store.directory(id).path))
    }
    func testWrongIdentityAndChecksumRejected() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try CompanionStore(root: root)
        let manifest = Dictionary(uniqueKeysWithValues: CompanionStore.files.map { ($0, ["size": 1, "sha256": digest(Data([1]))] as [String: Any]) })
        XCTAssertThrowsError(try store.begin(id: String(repeating: "a", count: 64), manifest: manifest))
        let id = digest(try JSONSerialization.data(withJSONObject: manifest, options: .sortedKeys))
        _ = try store.begin(id: id, manifest: manifest)
        for name in CompanionStore.files { _ = try store.append(id: id, name: name, offset: 0, data: Data([2])) }
        XCTAssertThrowsError(try store.commit(id))
        XCTAssertThrowsError(try store.readyDirectory(id))
    }
}
