import Foundation
import MLXPeerWorker

func printJSON<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    print(String(decoding: try encoder.encode(value), as: UTF8.self))
}

do {
    if CommandLine.arguments.count == 2, CommandLine.arguments[1] == "self-test" {
        let report = try WorkerSelfTest.run()
        try printJSON(report)
        if !report.passed { exit(1) }
    } else if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "run-fixture" {
        let result = try FixtureRunner.run(directory: URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true))
        try printJSON(result)
    } else {
        throw WorkerError.invalid("Usage: mlx-peer-stage run-fixture DIRECTORY | self-test")
    }
} catch {
    FileHandle.standardError.write(Data("mlx-peer-stage: \(error.localizedDescription)\n".utf8))
    exit(1)
}
