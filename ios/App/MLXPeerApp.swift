import SwiftUI
import UniformTypeIdentifiers
import MLXPeerWorker

@main
struct MLXPeerApp: App {
    var body: some Scene { WindowGroup { WorkerView() } }
}

@MainActor
final class WorkerViewModel: ObservableObject {
    @Published var busy = false
    @Published var result = "Ready. This build runs local feasibility probes."
    private let worker = DispatchQueue(label: "mlx-peer.stage", qos: .userInitiated)
    private var launchRequestHandled = false

    func selfTest() {
        run { try Self.encode(WorkerSelfTest.run()) }
    }

    func fixture(_ directory: URL) {
        run {
            let access = directory.startAccessingSecurityScopedResource()
            defer { if access { directory.stopAccessingSecurityScopedResource() } }
            return try Self.encode(FixtureRunner.run(directory: directory))
        }
    }

    /// Development instrumentation for Xcode/devicectl. Files stay in the app's Documents directory.
    func runLaunchRequest() {
        guard !launchRequestHandled else { return }
        launchRequestHandled = true
        let arguments = ProcessInfo.processInfo.arguments
        let selfTestRequested = arguments.contains("--run-self-test")
        let fixtureIndex = arguments.firstIndex(of: "--run-fixture")
        let serveIndex = arguments.firstIndex(of: "--serve-fixture")
        guard selfTestRequested || fixtureIndex != nil || serveIndex != nil else { return }
        guard [selfTestRequested, fixtureIndex != nil, serveIndex != nil].filter({ $0 }).count == 1 else {
            result = "Failed: choose only one launch probe."
            return
        }
        if let serveIndex {
            guard serveIndex + 1 < arguments.count else {
                result = "Failed: --serve-fixture requires a Documents subfolder."
                return
            }
            let name = arguments[serveIndex + 1]
            run {
                let directory = try FixtureRunner.member(name, root: Self.documentsDirectory())
                return try WireServer.run(directory: directory) { message in
                    DispatchQueue.main.async { self.result = message }
                }
            }
        } else if selfTestRequested {
            run {
                let json = try Self.encode(WorkerSelfTest.run())
                let documents = try Self.documentsDirectory()
                let destination = try FixtureRunner.member("self-test.json", root: documents)
                try Data(json.utf8).write(to: destination, options: .atomic)
                return json
            }
        } else if let fixtureIndex {
            guard fixtureIndex + 1 < arguments.count else {
                result = "Failed: --run-fixture requires one Documents subfolder name."
                return
            }
            let folderName = arguments[fixtureIndex + 1]
            run {
                let documents = try Self.documentsDirectory()
                let directory = try FixtureRunner.member(folderName, root: documents)
                guard try directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
                    throw WorkerError.invalid("Fixture argument must name a Documents subfolder")
                }
                return try Self.encode(FixtureRunner.run(directory: directory))
            }
        }
    }

    nonisolated private static func documentsDirectory() throws -> URL {
        try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                    appropriateFor: nil, create: true)
            .standardizedFileURL.resolvingSymlinksInPath()
    }

    private func run(_ operation: @escaping () throws -> String) {
        guard !busy else { return }
        busy = true
        UIApplication.shared.isIdleTimerDisabled = true
        result = "Computing on this device… Keep this app in the foreground."
        worker.async {
            let message: String
            do { message = try operation() }
            catch {
                message = "Failed: \(error.localizedDescription)"
                if let documents = try? Self.documentsDirectory() {
                    try? Data(message.utf8).write(to: documents.appendingPathComponent("last-error.txt"), options: .atomic)
                }
            }
            DispatchQueue.main.async { self.result = message; self.busy = false; UIApplication.shared.isIdleTimerDisabled = false }
        }
    }

    nonisolated private static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct WorkerView: View {
    @StateObject private var model = WorkerViewModel()
    @State private var importing = false
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("MLX on your iPhone").font(.largeTitle.bold())
                    Text("Execute assigned model layers on this iPhone and connect to your Mac over USB. Supports Qwen2 and quantized Qwen3.8 text layers.")
                        .foregroundStyle(.secondary)
                    Button("Run device self-test", action: model.selfTest)
                        .buttonStyle(.borderedProminent).disabled(model.busy)
                    Button("Open a Python reference fixture") { importing = true }
                        .buttonStyle(.bordered).disabled(model.busy)
                    Text("Choose the fixture folder in Files. Results are written into that folder for comparison on the Mac.")
                        .font(.footnote).foregroundStyle(.secondary)
                    if model.busy { ProgressView() }
                    Text(model.result).font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    Text("Foreground only · USB developer probe · No telemetry")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding()
            }.navigationTitle("MLX Peer")
        }
        .onAppear { model.runLaunchRequest() }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let url): model.fixture(url)
            case .failure(let error): model.result = error.localizedDescription
            }
        }
    }
}
