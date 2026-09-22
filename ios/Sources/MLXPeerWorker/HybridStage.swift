// Adapted from MLX-LM qwen3_5.py, qwen3_next.py and gated_delta.py.
// Copyright © 2023-2026 Apple Inc. MIT license: licenses/MLX-LM-MIT.txt.
import Foundation
import MLX

public protocol ModelStage: AnyObject {
    var hiddenSize: Int { get }
    var layerRange: Range<Int> { get }
    var maxContext: Int { get }
    var dtype: DType { get }
    var weightBytes: Int { get }
    var position: Int { get }
    func reset()
    func forward(_ input: MLXArray, position: Int) throws -> MLXArray
}

public struct HybridConfiguration: Decodable {
    struct Quantization: Decodable { let bits: Int; let group_size: Int; let mode: String }
    struct Rope: Decodable { let rope_theta: Float; let partial_rotary_factor: Float; let rope_type: String }
    struct Text: Decodable {
        let hidden_size: Int; let intermediate_size: Int; let num_hidden_layers: Int
        let num_attention_heads: Int; let num_key_value_heads: Int; let head_dim: Int
        let linear_num_value_heads: Int; let linear_num_key_heads: Int
        let linear_key_head_dim: Int; let linear_value_head_dim: Int; let linear_conv_kernel_dim: Int
        let full_attention_interval: Int; let rms_norm_eps: Float; let rope_parameters: Rope
        let attention_bias: Bool; let tie_word_embeddings: Bool; let hidden_act: String
        let num_experts: Int?
    }
    let model_type: String; let quantization: Quantization; let text_config: Text

    func validate() throws {
        let c = text_config
        guard model_type == "qwen3_5", quantization.bits == 4,
              quantization.group_size == 64, quantization.mode == "affine",
              c.hidden_size > 0, c.hidden_size <= 8192, c.hidden_size % 64 == 0,
              c.intermediate_size > 0, c.intermediate_size <= 32768, c.intermediate_size % 64 == 0,
              c.num_hidden_layers > 0, c.num_hidden_layers <= 128,
              c.num_attention_heads > 0, c.num_attention_heads <= 64,
              c.num_key_value_heads > 0, c.num_attention_heads % c.num_key_value_heads == 0,
              c.head_dim > 0, c.head_dim <= 256, c.head_dim % 64 == 0,
              c.linear_num_value_heads > 0, c.linear_num_value_heads <= 128,
              c.linear_num_key_heads > 0, c.linear_num_value_heads % c.linear_num_key_heads == 0,
              c.linear_key_head_dim > 0, c.linear_key_head_dim <= 256, c.linear_key_head_dim % 32 == 0,
              c.linear_value_head_dim > 0, c.linear_value_head_dim <= 256, c.linear_value_head_dim % 4 == 0,
              c.linear_conv_kernel_dim == 4, c.full_attention_interval == 4,
              !c.attention_bias, !c.tie_word_embeddings, c.hidden_act == "silu",
              (c.num_experts ?? 0) == 0,
              c.rope_parameters.rope_type == "default", c.rope_parameters.partial_rotary_factor == 0.25,
              c.rope_parameters.rope_theta.isFinite, c.rope_parameters.rope_theta > 0,
              c.rms_norm_eps.isFinite, c.rms_norm_eps > 0 else {
            throw WorkerError.invalid("Unsupported or out-of-bounds hybrid configuration")
        }
    }
}

public final class HybridStage: ModelStage {
    let c: HybridConfiguration.Text
    public var hiddenSize: Int { c.hidden_size }
    public let layerRange: Range<Int>
    public let maxContext: Int
    public let dtype: DType = .bfloat16
    public let weightBytes: Int
    public private(set) var position = 0
    private let weights: [String: MLXArray]
    private var states: [Int: MLXArray] = [:]
    private var convStates: [Int: MLXArray] = [:]
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]

    private static let silu: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { (x: MLXArray) in x * sigmoid(x) }
    private static let swiglu = compile(shapeless: true) { (x: [MLXArray]) in [x[0] * sigmoid(x[0]) * x[1]] }
    private static let preciseGate = compile(shapeless: true) { (x: [MLXArray]) -> [MLXArray] in
        let gate = x[1].asType(.float32)
        return [(gate * sigmoid(gate) * x[0].asType(.float32)).asType(x[0].dtype)]
    }
    private static let decay = compile(shapeless: true) { (x: [MLXArray]) -> [MLXArray] in
        [exp(-exp(x[0].asType(.float32)) * logAddExp(x[1] + x[2], 0)).asType(x[1].dtype)]
    }

    public init(configuration: HybridConfiguration, layerRange: Range<Int>, maxContext: Int,
                weights: [String: MLXArray]) throws {
        try configuration.validate()
        c = configuration.text_config
        guard !layerRange.isEmpty, layerRange.lowerBound >= 0, layerRange.upperBound <= c.num_hidden_layers,
              maxContext > 0, maxContext <= 8192 else { throw WorkerError.invalid("Invalid hybrid stage range/context") }
        self.layerRange = layerRange; self.maxContext = maxContext
        var expected = Set<String>()
        for i in layerRange {
            let prefix = "language_model.model.layers.\(i)."
            func tensor(_ suffix: String, _ shape: [Int], _ type: DType = .bfloat16) throws {
                let name = prefix + suffix
                guard let value = weights[name], value.shape == shape, value.dtype == type else {
                    throw WorkerError.invalid("Invalid hybrid weight \(name); expected \(shape), \(type)")
                }
                expected.insert(name)
            }
            func linear(_ name: String, _ input: Int, _ output: Int) throws {
                try tensor(name + ".weight", [output, input / 8], .uint32)
                try tensor(name + ".scales", [output, input / 64])
                try tensor(name + ".biases", [output, input / 64])
            }
            let h = c.hidden_size
            try tensor("input_layernorm.weight", [h]); try tensor("post_attention_layernorm.weight", [h])
            try linear("mlp.gate_proj", h, c.intermediate_size)
            try linear("mlp.up_proj", h, c.intermediate_size)
            try linear("mlp.down_proj", c.intermediate_size, h)
            if (i + 1) % 4 != 0 {
                let kd = c.linear_num_key_heads * c.linear_key_head_dim
                let vd = c.linear_num_value_heads * c.linear_value_head_dim
                try linear("linear_attn.in_proj_qkv", h, 2 * kd + vd)
                try linear("linear_attn.in_proj_z", h, vd)
                try linear("linear_attn.in_proj_a", h, c.linear_num_value_heads)
                try linear("linear_attn.in_proj_b", h, c.linear_num_value_heads)
                try linear("linear_attn.out_proj", vd, h)
                try tensor("linear_attn.conv1d.weight", [2 * kd + vd, 4, 1])
                try tensor("linear_attn.A_log", [c.linear_num_value_heads])
                try tensor("linear_attn.dt_bias", [c.linear_num_value_heads])
                try tensor("linear_attn.norm.weight", [c.linear_value_head_dim])
            } else {
                try linear("self_attn.q_proj", h, c.num_attention_heads * c.head_dim * 2)
                try linear("self_attn.k_proj", h, c.num_key_value_heads * c.head_dim)
                try linear("self_attn.v_proj", h, c.num_key_value_heads * c.head_dim)
                try linear("self_attn.o_proj", c.num_attention_heads * c.head_dim, h)
                try tensor("self_attn.q_norm.weight", [c.head_dim]); try tensor("self_attn.k_norm.weight", [c.head_dim])
            }
        }
        guard Set(weights.keys) == expected else { throw WorkerError.invalid("Unassigned hybrid weights") }
        self.weights = weights; weightBytes = weights.values.reduce(0) { $0 + $1.nbytes }
        Memory.cacheLimit = 64 * 1024 * 1024
        eval(Array(weights.values))
    }

    public func reset() { position = 0; states.removeAll(); convStates.removeAll(); keys.removeAll(); values.removeAll() }

    public func forward(_ input: MLXArray, position requested: Int) throws -> MLXArray {
        guard input.ndim == 3, input.dim(0) == 1, input.dim(2) == hiddenSize,
              input.dtype == dtype, input.dim(1) > 0, input.dim(1) <= maxContext - position,
              requested == position else { throw WorkerError.invalid("Hybrid shape/dtype/context/position mismatch; reset required") }
        let count = input.dim(1)
        var output = input
        for i in layerRange {
            // Opt-in local diagnostics; absent during phone and capacity runs.
            func trace(_ name: String, _ value: MLXArray) throws {
                #if os(macOS)
                if let path = ProcessInfo.processInfo.environment["MLX_PEER_TRACE_DIRECTORY"], requested == 0 {
                    try save(arrays: ["value": value], metadata: ["format": "mlx-peer-trace"],
                        url: URL(fileURLWithPath: path).appendingPathComponent("swift-\(i)-\(name).safetensors"))
                }
                #endif
            }
            let prefix = "language_model.model.layers.\(i)."
            func w(_ name: String) -> MLXArray { weights[prefix + name]! }
            func linear(_ x: MLXArray, _ name: String) -> MLXArray {
                quantizedMM(x, w(name + ".weight"), scales: w(name + ".scales"), biases: w(name + ".biases"),
                            transpose: true, groupSize: 64, bits: 4, mode: .affine)
            }
            let n = MLXFast.rmsNorm(output, weight: w("input_layernorm.weight"), eps: c.rms_norm_eps)
            try trace("norm", n)
            let attended: MLXArray
            if (i + 1) % 4 != 0 {
                let hk = c.linear_num_key_heads, hv = c.linear_num_value_heads
                let dk = c.linear_key_head_dim, dv = c.linear_value_head_dim
                let kd = hk * dk, vd = hv * dv, convDim = 2 * kd + vd
                let qkv = linear(n, "linear_attn.in_proj_qkv")
                try trace("qkv", qkv)
                let z = linear(n, "linear_attn.in_proj_z").reshaped(1, count, hv, dv)
                let a = linear(n, "linear_attn.in_proj_a"), b = linear(n, "linear_attn.in_proj_b")
                let previous = convStates[i] ?? MLXArray.zeros([1, 3, convDim], dtype: dtype)
                let convInput = concatenated([previous, qkv], axis: 1)
                convStates[i] = convInput[0..., (convInput.dim(1)-3)..., 0...]
                let conv = Self.silu(conv1d(convInput, w("linear_attn.conv1d.weight"), groups: convDim))
                try trace("conv", conv)
                let parts = split(conv, indices: [kd, 2 * kd], axis: -1)
                let rawQ = parts[0].reshaped(1, count, hk, dk)
                let rawK = parts[1].reshaped(1, count, hk, dk)
                let v = parts[2].reshaped(1, count, hv, dv)
                let ones = MLXArray.ones([dk], dtype: dtype)
                let inv: Float = 1 / sqrt(Float(dk))
                let q = (inv * inv) * MLXFast.rmsNorm(rawQ, weight: ones, eps: 1e-6)
                let k = inv * MLXFast.rmsNorm(rawK, weight: ones, eps: 1e-6)
                let g = Self.decay([w("linear_attn.A_log"), a, w("linear_attn.dt_bias")])[0]
                let beta = sigmoid(b)
                try trace("q", q); try trace("k", k); try trace("g", g); try trace("beta", beta)
                let state = states[i] ?? MLXArray.zeros([1, hv, dv, dk], dtype: dtype)
                let recurrent = HybridDeltaKernel.run(q: q, k: k, v: v, g: g, beta: beta, state: state)
                states[i] = recurrent.1
                try trace("recurrent", recurrent.0)
                let norm = MLXFast.rmsNorm(recurrent.0, weight: w("linear_attn.norm.weight"), eps: c.rms_norm_eps)
                let gated = Self.preciseGate([norm, z])[0].reshaped(1, count, vd)
                try trace("gated", gated)
                attended = linear(gated, "linear_attn.out_proj")
            } else {
                let parts = split(linear(n, "self_attn.q_proj").reshaped(1, count, c.num_attention_heads, 2 * c.head_dim), parts: 2, axis: -1)
                let gate = parts[1].reshaped(1, count, -1)
                var q = MLXFast.rmsNorm(parts[0], weight: w("self_attn.q_norm.weight"), eps: c.rms_norm_eps).transposed(0, 2, 1, 3)
                var k = MLXFast.rmsNorm(linear(n, "self_attn.k_proj").reshaped(1, count, c.num_key_value_heads, c.head_dim), weight: w("self_attn.k_norm.weight"), eps: c.rms_norm_eps).transposed(0, 2, 1, 3)
                var v = linear(n, "self_attn.v_proj").reshaped(1, count, c.num_key_value_heads, c.head_dim).transposed(0, 2, 1, 3)
                let dimensions = Int(Float(c.head_dim) * c.rope_parameters.partial_rotary_factor)
                q = MLXFast.RoPE(q, dimensions: dimensions, traditional: false, base: c.rope_parameters.rope_theta, scale: 1, offset: position)
                k = MLXFast.RoPE(k, dimensions: dimensions, traditional: false, base: c.rope_parameters.rope_theta, scale: 1, offset: position)
                try trace("qrot", q); try trace("krot", k); try trace("v", v); try trace("gate", gate)
                if let pk = keys[i], let pv = values[i] { k = concatenated([pk, k], axis: 2); v = concatenated([pv, v], axis: 2) }
                keys[i] = k; values[i] = v
                let result = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(c.head_dim)), mask: .causal)
                    .transposed(0, 2, 1, 3).reshaped(1, count, -1)
                try trace("attention", result)
                attended = linear(result * sigmoid(gate), "self_attn.o_proj")
            }
            output = output + attended
            try trace("attended", attended)
            let post = MLXFast.rmsNorm(output, weight: w("post_attention_layernorm.weight"), eps: c.rms_norm_eps)
            let mlp = Self.swiglu([linear(post, "mlp.gate_proj"), linear(post, "mlp.up_proj")])[0]
            output = output + linear(mlp, "mlp.down_proj")
            try trace("output", output)
            eval([output] + [states[i], convStates[i], keys[i], values[i]].compactMap { $0 })
        }
        position += count
        return output
    }
}
