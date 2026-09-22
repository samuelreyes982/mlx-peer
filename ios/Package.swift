// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "MLXPeerWorker",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MLXPeerWorker", targets: ["MLXPeerWorker"]),
        .executable(name: "mlx-peer-stage", targets: ["MLXPeerStage"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6")
    ],
    targets: [
        .target(name: "MLXPeerWorker", dependencies: [.product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "MLXPeerStage", dependencies: ["MLXPeerWorker"]),
        .testTarget(name: "MLXPeerWorkerTests", dependencies: ["MLXPeerWorker"]),
    ],
    swiftLanguageModes: [.v5]
)
