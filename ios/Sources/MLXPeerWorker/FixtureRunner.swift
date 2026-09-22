import Foundation
import MLX
#if os(iOS)
import os
#endif

public struct FixtureRequest: Decodable {
    public let protocolVersion: Int
    public let layerStart: Int
    public let layerEnd: Int
    public let sessionID: String
    public let position: Int
    public let maxContext: Int
    public let steps: [FixtureStep]?
    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version", layerStart = "layer_start"
        case layerEnd = "layer_end", sessionID = "session_id", position
        case maxContext = "max_context", steps
    }
}

public struct FixtureStep: Decodable {
    public let input: String
    public let output: String
    public let position: Int
    public init(input: String, output: String, position: Int) {
        self.input = input; self.output = output; self.position = position
    }
}

public struct StepReport: Codable {
    public let input: String
    public let output: String
    public let position: Int
    public let shape: [Int]
    public let dtype: String
    public let computeMilliseconds: Double
    public let mlxActiveBytes: Int
    public let mlxPeakBytes: Int
}

public struct FixtureReport: Codable {
    public let protocolVersion: Int
    public let sessionID: String
    public let runtime: String
    public let system: String
    public let layerStart: Int
    public let layerEnd: Int
    public let weightBytes: Int
    public let steps: [StepReport]
    public let deviceIdentityVerified: Bool
}

public enum FixtureRunner {
    /// Load only the pre-provisioned, validated local stage.
    public static func loadStage(directory: URL) throws -> (FixtureRequest, any ModelStage) {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let requestURL = try member("request.json", root: root)
        let request = try JSONDecoder().decode(FixtureRequest.self, from: boundedRead(requestURL, limit: 65536))
        guard request.protocolVersion == 1, !request.sessionID.isEmpty,
              request.sessionID.utf8.count <= 128, request.position == 0,
              request.layerStart >= 0, request.layerEnd > request.layerStart else {
            throw WorkerError.invalid("Unsupported protocol, invalid session/layers, or nonzero initial position")
        }
        let configData = try boundedRead(try member("config.json", root: root), limit: 65536)
        let weightURL = try member("weights.safetensors", root: root)
        let modelType = (try JSONSerialization.jsonObject(with: configData) as? [String: Any])?["model_type"] as? String
        if modelType == "qwen3_5" {
            try checkSize(weightURL, limit: 3 * 1024 * 1024 * 1024)
            #if os(iOS)
            let bytes = (try FileManager.default.attributesOfItem(atPath: weightURL.path)[.size] as! NSNumber).intValue
            guard bytes + 384 * 1024 * 1024 < Int(os_proc_available_memory()) else {
                throw WorkerError.invalid("Insufficient process headroom for hybrid weights plus reserve")
            }
            #endif
            let config = try JSONDecoder().decode(HybridConfiguration.self, from: configData)
            return (request, try HybridStage(configuration: config, layerRange: request.layerStart..<request.layerEnd,
                maxContext: request.maxContext, weights: loadArrays(url: weightURL)))
        }
        let config = try JSONDecoder().decode(QwenConfiguration.self, from: configData)
        try checkSize(weightURL, limit: 1_073_741_824)
        let stage = try QwenStage(configuration: config,
            layerRange: request.layerStart..<request.layerEnd, maxContext: request.maxContext,
            weights: loadArrays(url: weightURL))
        return (request, stage)
    }

    public static func run(directory: URL) throws -> FixtureReport {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let (request, stage) = try loadStage(directory: root)
        let steps = request.steps ?? [.init(input: "input.safetensors", output: "output.safetensors", position: request.position)]
        guard !steps.isEmpty, steps.count <= 4096 else { throw WorkerError.invalid("Invalid step count") }
        var reserved: Set<String> = ["config.json", "request.json", "weights.safetensors", "report.json"]
        for step in steps { reserved.insert(step.input) }
        var outputs = Set<String>()
        for step in steps {
            guard !reserved.contains(step.output), outputs.insert(step.output).inserted,
                  step.input.hasSuffix(".safetensors"), step.output.hasSuffix(".safetensors") else {
                throw WorkerError.invalid("Output overlaps fixture inputs, another output, or has wrong extension")
            }
            _ = try member(step.input, root: root)
            _ = try member(step.output, root: root)
        }
        var reports: [StepReport] = []
        for step in steps {
            let inputURL = try member(step.input, root: root)
            try checkSize(inputURL, limit: 67_108_864)
            let tensors = try loadArrays(url: inputURL)
            guard Set(tensors.keys) == ["hidden_states"], let input = tensors["hidden_states"] else {
                throw WorkerError.invalid("Input file must contain only hidden_states")
            }
            eval(input)
            let start = ProcessInfo.processInfo.systemUptime
            let output = try stage.forward(input, position: step.position)
            let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
            try save(arrays: ["hidden_states": output], metadata: ["format": "mlx-peer-v1"],
                     url: member(step.output, root: root))
            reports.append(.init(input: step.input, output: step.output, position: step.position,
                shape: output.shape, dtype: String(describing: output.dtype), computeMilliseconds: elapsed,
                mlxActiveBytes: Memory.activeMemory, mlxPeakBytes: Memory.peakMemory))
        }
        let report = FixtureReport(protocolVersion: 1, sessionID: request.sessionID,
            runtime: "mlx-swift 0.31.6 (MLX 0.31.1)",
            system: ProcessInfo.processInfo.operatingSystemVersionString,
            layerStart: request.layerStart, layerEnd: request.layerEnd,
            weightBytes: stage.weightBytes, steps: reports, deviceIdentityVerified: false)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: member("report.json", root: root), options: .atomic)
        return report
    }

    public static func member(_ name: String, root: URL) throws -> URL {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\\"),
              !name.contains("\0"), name.utf8.count <= 255 else {
            throw WorkerError.invalid("Fixture paths must be single local filenames")
        }
        let candidate = root.appendingPathComponent(name).standardizedFileURL
        let resolved = candidate.resolvingSymlinksInPath().standardizedFileURL
        guard resolved.deletingLastPathComponent().path == root.path,
              candidate.path == resolved.path else {
            throw WorkerError.invalid("Fixture members must not be symbolic links")
        }
        return resolved
    }

    private static func checkSize(_ url: URL, limit: Int) throws {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attrs[.type] as? FileAttributeType == .typeRegular,
              let size = attrs[.size] as? NSNumber, size.intValue > 0, size.intValue <= limit else {
            throw WorkerError.invalid("File is empty, not regular, or exceeds prototype size budget: \(url.lastPathComponent)")
        }
    }
    private static func boundedRead(_ url: URL, limit: Int) throws -> Data {
        try checkSize(url, limit: limit)
        return try Data(contentsOf: url)
    }
}
