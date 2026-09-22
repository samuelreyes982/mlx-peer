import XCTest
@testable import MLXPeerWorker

final class ValidationTests: XCTestCase {
    func testPathTraversalRejected() throws {
        let root = URL(fileURLWithPath: "/tmp/fixture", isDirectory: true)
        for name in ["../weights.safetensors", "/tmp/weights.safetensors", "a/b", "..", ""] {
            XCTAssertThrowsError(try FixtureRunner.member(name, root: root))
        }
        XCTAssertEqual(try FixtureRunner.member("input.safetensors", root: root).lastPathComponent, "input.safetensors")
    }

    func testUnsupportedArchitectureRejected() throws {
        let data = Data("{\"model_type\":\"qwen3\"}".utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(QwenConfiguration.self, from: data))
    }

    func testSymlinkCannotAliasAReservedFile() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("weights.safetensors")
        try Data("fixture".utf8).write(to: original)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("output.safetensors"),
                                                  withDestinationURL: original)
        XCTAssertThrowsError(try FixtureRunner.member("output.safetensors", root: root))
    }
}
