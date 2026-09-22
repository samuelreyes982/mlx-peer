import Foundation
import MLX
#if os(iOS)
import os
#endif

public struct SelfTestReport: Codable {
    public let passed: Bool
    public let maximumAbsoluteDifference: Float
    public let tolerance: Float
    public let elapsedMilliseconds: Double
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int
    /// Current process headroom after the probe, not installed RAM or a promised allocation budget.
    public let availableProcessMemoryBytes: Int?
    public let runtime: String
    public let system: String
    public let scope: String
}

public enum WorkerSelfTest {
    /// Checks full-prefill versus cache continuation using tiny deterministic dense weights.
    /// Independent Python/Swift parity is established by imported fixtures, not by this test.
    public static func run() throws -> SelfTestReport {
        let start = ProcessInfo.processInfo.systemUptime
        let json = """
        {"model_type":"qwen2","hidden_size":32,"intermediate_size":64,
         "num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":1,
         "rms_norm_eps":0.000001,"rope_theta":1000000}
        """
        let config = try JSONDecoder().decode(QwenConfiguration.self, from: Data(json.utf8))
        let shapes = [
            "input_layernorm.weight": [32], "post_attention_layernorm.weight": [32],
            "self_attn.q_proj.weight": [32,32], "self_attn.q_proj.bias": [32],
            "self_attn.k_proj.weight": [16,32], "self_attn.k_proj.bias": [16],
            "self_attn.v_proj.weight": [16,32], "self_attn.v_proj.bias": [16],
            "self_attn.o_proj.weight": [32,32],
            "mlp.gate_proj.weight": [64,32], "mlp.up_proj.weight": [64,32],
            "mlp.down_proj.weight": [32,64],
        ]
        var weights: [String: MLXArray] = [:]
        for (index, entry) in shapes.sorted(by: {$0.key < $1.key}).enumerated() {
            let (name, shape) = entry
            let count = shape.reduce(1, *)
            let data: [Float] = (0..<count).map {
                name.contains("layernorm") ? 1 : Float((($0 * 13 + index * 7) % 29) - 14) / 300
            }
            weights["model.layers.0." + name] = MLXArray(data, shape)
        }
        let inputData: [Float] = (0..<5*32).map { Float(($0 * 3) % 17 - 8) / 10 }
        let input = MLXArray(inputData, [1,5,32])
        let stage = try QwenStage(configuration: config, layerRange: 0..<1, maxContext: 16, weights: weights)
        let full = try stage.forward(input, position: 0)
        stage.reset()
        let prefix = try stage.forward(MLXArray(Array(inputData.prefix(3*32)), [1,3,32]), position: 0)
        let suffix = try stage.forward(MLXArray(Array(inputData.suffix(2*32)), [1,2,32]), position: 3)
        let difference = abs(full - concatenated([prefix, suffix], axis: 1)).max().item(Float.self)
        let tolerance: Float = 0.0001
        #if os(iOS)
        let availableBytes: Int? = Int(os_proc_available_memory())
        #else
        let availableBytes: Int? = nil
        #endif
        return SelfTestReport(passed: difference.isFinite && difference <= tolerance,
            maximumAbsoluteDifference: difference, tolerance: tolerance,
            elapsedMilliseconds: (ProcessInfo.processInfo.systemUptime - start) * 1000,
            mlxActiveBytes: Memory.activeMemory,
            mlxCacheBytes: Memory.cacheMemory,
            mlxPeakBytes: Memory.peakMemory,
            availableProcessMemoryBytes: availableBytes,
            runtime: "mlx-swift 0.31.6 (MLX 0.31.1)",
            system: ProcessInfo.processInfo.operatingSystemVersionString,
            scope: "Tiny Qwen2 cache consistency only. No network, no capacity claim, no independent reference.")
    }
}
