import SwiftUI
import UniformTypeIdentifiers
import MLXPeerWorker

@main
struct MLXPeerApp: App {
    var body: some Scene {
        WindowGroup {
            #if DEBUG
            if ProcessInfo.processInfo.arguments.contains(where: { ["--run-self-test", "--run-fixture", "--serve-fixture"].contains($0) }) {
                WorkerView()
            } else { CompanionView() }
            #else
            CompanionView()
            #endif
        }
    }
}

@MainActor
final class CompanionViewModel: ObservableObject {
    @Published var active = false
    @Published var stopping = false
    @Published var paired = false
    @Published var code = ""
    @Published var status = "Connect your iPhone to your Mac with a USB cable."
    private var server: CompanionServer?
    private var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Companion", isDirectory: true)
    }
    func start() {
        guard !active, !stopping else { return }
        do {
            let service = try CompanionServer(root: root)
            server = service; code = service.pairingCode; paired = service.isPaired; active = true
            UIApplication.shared.isIdleTimerDisabled = true
            #if DEBUG
            if ProcessInfo.processInfo.arguments.contains("--companion-test-report") {
                let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("companion-test-pairing.json")
                let data = try JSONSerialization.data(withJSONObject: ["code": service.pairingCode, "peer_id": service.peerID, "paired": service.isPaired])
                try data.write(to: url, options: [.atomic, .completeFileProtection])
            }
            #endif
            DispatchQueue.global(qos: .userInitiated).async {
                var failure: String?
                do {
                    try service.run { message in
                        DispatchQueue.main.async { self.status = message; self.paired = service.isPaired }
                    }
                } catch { failure = error.localizedDescription }
                let result = failure
                DispatchQueue.main.async {
                    self.active = false; self.stopping = false
                    UIApplication.shared.isIdleTimerDisabled = false
                    if let result { self.status = result }
                }
            }
        } catch { status = error.localizedDescription; active = false; UIApplication.shared.isIdleTimerDisabled = false }
    }
    func stop() {
        guard active else { return }
        stopping = true; status = "Stopping sharing…"; server?.stop()
        UIApplication.shared.isIdleTimerDisabled = false
    }
    func forget() {
        guard !active, !stopping else { return }
        do {
            let service = try CompanionServer(root: root)
            try service.forgetPairing(); server = service; paired = false; code = service.pairingCode
            status = "Previous Mac forgotten. Start sharing to show a new code."
        } catch { status = error.localizedDescription }
    }
    func removeModels() {
        guard !active, !stopping else { return }
        do { try CompanionServer(root: root).removeModels(); status = "Saved models removed from this iPhone." }
        catch { status = error.localizedDescription }
    }
}

struct CompanionView: View {
    @StateObject private var model = CompanionViewModel()
    @Environment(\.scenePhase) private var scenePhase
    @State private var confirmRemoval = false
    @State private var confirmForget = false
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 26) {
                    Image("PeerMark").resizable().frame(width: 80, height: 80)
                        .clipShape(RoundedRectangle(cornerRadius: 20)).accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Local AI.\nShared power.").font(.largeTitle.bold())
                        Text("Your Mac and iPhone share a local model. Your iPhone runs part of the computation over USB.").foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 16) {
                        Label(model.paired ? "Mac paired" : "Pair your Mac", systemImage: model.paired ? "checkmark.shield.fill" : "link").font(.headline)
                        if model.active && !model.paired {
                            Text(model.code).font(.system(size: 44, weight: .semibold, design: .monospaced)).tracking(6).minimumScaleFactor(0.6).lineLimit(1).accessibilityLabel("Pairing code \(model.code)")
                            Text("Enter this code in MLX Peer on your Mac. The code expires after five minutes; restart sharing for a new code.").font(.footnote).foregroundStyle(.secondary)
                        }
                        Text(model.status).font(.callout).accessibilityIdentifier("connectionStatus")
                        Button(model.active ? "Stop sharing" : "Start sharing") { model.active ? model.stop() : model.start() }
                            .buttonStyle(.borderedProminent).disabled(model.stopping)
                    }.padding(20).frame(maxWidth: .infinity, alignment: .leading).background(.blue.opacity(0.07), in: RoundedRectangle(cornerRadius: 20))
                    VStack(alignment: .leading, spacing: 12) {
                        Label("Keep this app open and the phone unlocked.", systemImage: "sun.max")
                        Label("Choose your model on the Mac. Transfers are automatic.", systemImage: "folder")
                        Label("No cloud inference or account required.", systemImage: "lock.shield")
                    }.font(.subheadline).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 12) {
                        NavigationLink { SetupGuideView() } label: { Label("Set up your Mac & iPhone", systemImage: "cable.connector") }
                        NavigationLink { PrivacyView() } label: { Label("Privacy & data", systemImage: "hand.raised") }
                        Button("Pair a new Mac") { confirmForget = true }.disabled(model.active || model.stopping)
                        Button("Remove saved models", role: .destructive) { confirmRemoval = true }.disabled(model.active || model.stopping)
                        Text("Stop sharing to manage pairing and storage. Qwen2 / Qwen2.5 · FP16").font(.caption).foregroundStyle(.secondary)
                        Text("Version \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(24)
            }.navigationTitle("MLX Peer")
                .toolbar { ToolbarItem(placement: .topBarTrailing) { NavigationLink("Help") { SetupGuideView() } } }
        }
        .onAppear { model.start() }
        .onChange(of: scenePhase) { _, phase in if phase == .background { model.stop() } }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in model.stop() }
        .confirmationDialog("Forget the current Mac?", isPresented: $confirmForget, titleVisibility: .visible) { Button("Forget Mac", role: .destructive) { model.forget() } }
        .confirmationDialog("Remove saved model files? Your Mac's original files stay on your Mac.", isPresented: $confirmRemoval, titleVisibility: .visible) { Button("Remove models", role: .destructive) { model.removeModels() } }
    }
}

struct SetupGuideView: View {
    var body: some View {
        List {
            Section("One model. Two devices.") {
                Text("MLX Peer is a wired companion for an Apple Silicon Mac. Chat on your Mac; this iPhone runs the model layers assigned to it. A Mac, its companion app, and a USB data cable are required.")
            }
            Section("1 · Get the Mac app") {
                Text("Download MLX Peer for Apple Silicon from the project's GitHub releases. macOS 14 or later is required.")
                Link("Mac downloads & installation", destination: URL(string: "https://github.com/samuelreyes982/mlx-peer/releases")!)
            }
            Section("2 · Connect with USB") {
                Text("Connect this iPhone directly to your Mac with a USB data cable. Unlock the phone and accept Apple's Trust This Computer prompt if it appears. Keep MLX Peer open on the phone while sharing. Wi-Fi pairing is not supported.")
            }
            Section("3 · Pair once") {
                Text("Start sharing on this iPhone. In the Mac app, select this phone and enter the six-digit code shown here. Pairing is remembered until you choose Pair a new Mac. Only pair with a Mac you trust.")
            }
            Section("4 · Choose a local model") {
                Text("On your Mac, choose a supported Hugging Face model folder. Start with Qwen2.5-0.5B-Instruct. The companion supports dense Qwen2 and Qwen2.5 checkpoints prepared as FP16. GGUF and quantized checkpoints are not supported in this flow.")
                Text("Your Mac sends the assigned weights over USB, checks their integrity, and runs inference across both devices. Models are downloaded separately; no model is included in this app.")
            }
            Section("If it doesn't connect") {
                Text("Check that your cable carries data, the phone is unlocked, and this app is open. Restart sharing if the code has expired. If the phone gets too warm or runs low on memory, stop sharing, let it cool, and select fewer phone layers or a smaller model on your Mac.")
                Text("A USB-C connector does not make an iPhone a Thunderbolt device. Cable and device speeds vary. MLX Peer does not promise faster inference, lower power use, or support for every model.")
                Link("Setup guide & troubleshooting", destination: URL(string: "https://github.com/samuelreyes982/mlx-peer/blob/main/docs/SUPPORT.md")!)
            }
        }.navigationTitle("Get connected").navigationBarTitleDisplayMode(.inline)
    }
}

struct PrivacyView: View {
    var body: some View {
        List {
            Section("Your devices. Your computation.") {
                Text("MLX Peer by Samuel Reyes does not send your prompts, model files, or computation to the developer or an AI cloud service. There are no ads, analytics SDKs, tracking, or MLX Peer accounts.")
            }
            Section("What the iPhone stores") {
                Text("The app stores a random pairing credential, a device identifier used for that pairing, and model files received from your Mac. They stay in this app's storage. Model and pairing storage are excluded from iCloud backup.")
                Text("During inference, intermediate numerical results travel between your Mac and iPhone over USB. Treat these as potentially sensitive. Pair only with your own trusted Mac. The app does not add end-to-end encryption to the USB transport.")
            }
            Section("You control sharing") {
                Text("Stop sharing closes the connection. Backgrounding or locking the phone also stops sharing. After stopping, use Pair a new Mac to revoke pairing or Remove saved models to delete weights from this phone. Deleting the iPhone app removes its local data. Your Mac's files are managed separately.")
            }
            Section("External links & support") {
                Text("Opening GitHub or a model provider uses that service's privacy practices. Information you voluntarily post in a public GitHub issue is public; never include prompts, pairing codes, credentials, or personal files. Apple may handle diagnostics according to your device's sharing settings.")
                Link("Full privacy policy", destination: URL(string: "https://github.com/samuelreyes982/mlx-peer/blob/main/docs/PRIVACY.md")!)
                Link("Support", destination: URL(string: "https://github.com/samuelreyes982/mlx-peer/blob/main/docs/SUPPORT.md")!)
                NavigationLink("Open-source acknowledgements") { AcknowledgementsView() }
            }
        }.navigationTitle("Privacy & data").navigationBarTitleDisplayMode(.inline)
    }
}

struct AcknowledgementsView: View {
    private var text: String {
        guard let url = Bundle.main.url(forResource: "Acknowledgements", withExtension: "txt"),
              let content = try? String(contentsOf: url, encoding: .utf8) else { return "MLX Peer uses MLX and MLX Swift. See the project repository for third-party licenses." }
        return content
    }
    var body: some View {
        ScrollView { Text(text).font(.footnote).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding() }
            .navigationTitle("Acknowledgements").navigationBarTitleDisplayMode(.inline)
    }
}

#if DEBUG
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
#endif
