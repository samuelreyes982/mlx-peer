import Foundation
import MLX

public enum WorkerError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

/// Dense Qwen2/Qwen2.5 only. Unsupported architecture options fail before MLX allocation.
public struct QwenConfiguration: Decodable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let heads: Int
    public let kvHeads: Int
    public let layers: Int
    public let epsilon: Float
    public let ropeTheta: Float
    public var headDimension: Int { hiddenSize / heads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size", heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads", layers = "num_hidden_layers"
        case epsilon = "rms_norm_eps", ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling", sliding = "use_sliding_window"
        case hiddenActivation = "hidden_act", headDimension = "head_dim"
        case quantization, ropeTraditional = "rope_traditional"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard try c.decode(String.self, forKey: .modelType) == "qwen2" else {
            throw WorkerError.invalid("Only dense model_type=qwen2 is supported")
        }
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        layers = try c.decode(Int.self, forKey: .layers)
        epsilon = try c.decode(Float.self, forKey: .epsilon)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        guard hiddenSize > 0, hiddenSize <= 16384, intermediateSize > 0,
              intermediateSize <= 131072, heads > 0, kvHeads > 0,
              hiddenSize % heads == 0, heads % kvHeads == 0,
              hiddenSize / heads % 2 == 0, layers > 0, layers <= 128,
              epsilon.isFinite, epsilon > 0, ropeTheta.isFinite, ropeTheta > 0 else {
            throw WorkerError.invalid("Invalid or out-of-bounds Qwen dimensions")
        }
        if c.contains(.ropeScaling), try !c.decodeNil(forKey: .ropeScaling) {
            throw WorkerError.invalid("RoPE scaling is not supported in this feasibility build")
        }
        if c.contains(.quantization), try !c.decodeNil(forKey: .quantization) {
            throw WorkerError.invalid("Quantized weights are not supported in this feasibility build")
        }
        let sliding = try c.decodeIfPresent(Bool.self, forKey: .sliding) ?? false
        let traditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        let activation = try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"
        let explicitHeadDimension = try c.decodeIfPresent(Int.self, forKey: .headDimension) ?? headDimension
        guard !sliding, !traditional, activation == "silu", explicitHeadDimension == headDimension else {
            throw WorkerError.invalid("Unsupported attention or activation configuration")
        }
    }
}

/// Single-owner, synchronous stage. Call on one serial executor; MLX arrays never cross actors.
public final class QwenStage: ModelStage {
    public var hiddenSize: Int { configuration.hiddenSize }
    public let configuration: QwenConfiguration
    public let layerRange: Range<Int>
    public let maxContext: Int
    public let dtype: DType
    public let weightBytes: Int
    public private(set) var position = 0
    private let weights: [String: MLXArray]
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]

    public init(configuration: QwenConfiguration, layerRange: Range<Int>,
                maxContext: Int, weights: [String: MLXArray]) throws {
        guard !layerRange.isEmpty, layerRange.lowerBound >= 0,
              layerRange.upperBound <= configuration.layers,
              maxContext > 0, maxContext <= 4096 else {
            throw WorkerError.invalid("Invalid layer interval or context limit (1...4096)")
        }
        let h = configuration.hiddenSize
        let kv = configuration.kvHeads * configuration.headDimension
        let i = configuration.intermediateSize
        let shapes = [
            "input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
            "self_attn.q_proj.weight": [h, h], "self_attn.q_proj.bias": [h],
            "self_attn.k_proj.weight": [kv, h], "self_attn.k_proj.bias": [kv],
            "self_attn.v_proj.weight": [kv, h], "self_attn.v_proj.bias": [kv],
            "self_attn.o_proj.weight": [h, h],
            "mlp.gate_proj.weight": [i, h], "mlp.up_proj.weight": [i, h],
            "mlp.down_proj.weight": [h, i],
        ]
        var expected = Set<String>()
        var selectedDType: DType?
        var bytes = 0
        for layer in layerRange {
            for (suffix, shape) in shapes {
                let key = "model.layers.\(layer).\(suffix)"
                expected.insert(key)
                guard let value = weights[key], value.shape == shape else {
                    throw WorkerError.invalid("Missing or incorrectly shaped weight: \(key), expected \(shape)")
                }
                guard value.dtype == .float16 || value.dtype == .float32 else {
                    throw WorkerError.invalid("Weights must be dense float16 or float32: \(key)")
                }
                if let selectedDType, value.dtype != selectedDType {
                    throw WorkerError.invalid("All stage weights must use the same dtype")
                }
                selectedDType = value.dtype
                bytes += value.nbytes
            }
        }
        guard Set(weights.keys) == expected else {
            throw WorkerError.invalid("Shard includes unassigned or unsupported weights")
        }
        self.configuration = configuration
        self.layerRange = layerRange
        self.maxContext = maxContext
        self.weights = weights
        self.dtype = selectedDType!
        self.weightBytes = bytes
        eval(Array(weights.values))
    }

    public func reset() { keys.removeAll(); values.removeAll(); position = 0 }

    public func forward(_ input: MLXArray, position requestedPosition: Int) throws -> MLXArray {
        guard input.ndim == 3, input.dim(0) == 1,
              input.dim(1) > 0, input.dim(2) == configuration.hiddenSize,
              input.dtype == dtype else {
            throw WorkerError.invalid("Input must be [1, tokens, hidden_size] with the shard dtype")
        }
        let count = input.dim(1)
        guard requestedPosition == position else {
            throw WorkerError.invalid("Cache position mismatch: expected \(position), received \(requestedPosition); reset instead of retrying")
        }
        guard count <= maxContext - position else {
            throw WorkerError.invalid("Request exceeds context budget")
        }
        let c = configuration
        var output = input
        var nextKeys = keys
        var nextValues = values
        for layer in layerRange {
            let prefix = "model.layers.\(layer)."
            func w(_ suffix: String) -> MLXArray { weights[prefix + suffix]! }
            func linear(_ x: MLXArray, _ name: String, bias: Bool = false) -> MLXArray {
                let projected = matmul(x, w(name + ".weight").T)
                return bias ? projected + w(name + ".bias") : projected
            }
            let n = MLXFast.rmsNorm(output, weight: w("input_layernorm.weight"), eps: c.epsilon)
            var q = linear(n, "self_attn.q_proj", bias: true)
                .reshaped(1, count, c.heads, c.headDimension).transposed(0, 2, 1, 3)
            var k = linear(n, "self_attn.k_proj", bias: true)
                .reshaped(1, count, c.kvHeads, c.headDimension).transposed(0, 2, 1, 3)
            var v = linear(n, "self_attn.v_proj", bias: true)
                .reshaped(1, count, c.kvHeads, c.headDimension).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: c.headDimension, traditional: false,
                            base: c.ropeTheta, scale: 1, offset: position)
            k = MLXFast.RoPE(k, dimensions: c.headDimension, traditional: false,
                            base: c.ropeTheta, scale: 1, offset: position)
            if let previousKeys = keys[layer], let previousValues = values[layer] {
                k = concatenated([previousKeys, k], axis: 2)
                v = concatenated([previousValues, v], axis: 2)
            }
            nextKeys[layer] = k
            nextValues[layer] = v
            // MLX causal masking aligns queries to the final positions of the KV sequence.
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: 1 / sqrt(Float(c.headDimension)), mask: .causal)
                .transposed(0, 2, 1, 3).reshaped(1, count, c.hiddenSize)
            output = output + linear(attended, "self_attn.o_proj")
            let post = MLXFast.rmsNorm(output, weight: w("post_attention_layernorm.weight"), eps: c.epsilon)
            let gate = linear(post, "mlp.gate_proj")
            output = output + linear((gate * sigmoid(gate)) * linear(post, "mlp.up_proj"), "mlp.down_proj")
        }
        // Materialize before committing cache progression or crossing a transport boundary.
        eval([output] + Array(nextKeys.values) + Array(nextValues.values))
        keys = nextKeys
        values = nextValues
        position += count
        return output
    }
}
