import MoldClient
import SwiftUI
import VisionKit

/// Scan a Pairing Code: the QR from the Mac's Pair a Phone…, redeemed in
/// place. The Camera app opens the same code's universal link
/// (`PairingLinkSheet`); this app never registers `mold://`, which the Tauri
/// iPhone app owns, so older `mold://pair` codes are read here or pasted.
struct PairingScanView: View {
    @Environment(HostStore.self) private var hosts
    @Environment(CompanionStores.self) private var stores
    let done: () -> Void

    @State private var phase: Phase = .scanning
    @State private var pasted = ""
    @State private var camera = CameraAccess()
    @Environment(\.scenePhase) private var scenePhase

    enum Phase: Equatable {
        case scanning
        case confirming(MobilePairingPayload)
        case pairing(String)
        case failed(String)

        static func afterScan(_ payload: MobilePairingPayload) -> Self {
            payload.mode == .regular ? .confirming(payload) : .pairing(payload.name)
        }
    }

    var body: some View {
        List {
            Section {
                scanner
                    .frame(minHeight: 280)
                    .listRowInsets(EdgeInsets())
                    .clipShape(.rect(cornerRadius: 10))
            } footer: {
                Text("On your Mac, open Machines ▸ your machine ▸ Pair a Phone…")
                    .foregroundStyle(.secondaryText)
            }
            switch phase {
            case .scanning: EmptyView()
            case let .confirming(payload):
                Section {
                    Text(payload.name).font(.headline)
                    Text("\(payload.mode.title): \(payload.mode.explanation)").foregroundStyle(.secondaryText)
                    Text(payload.baseURL).font(.body.monospaced()).textSelection(.enabled)
                    Button("Pair") { pair(payload) }
                        .disabled(payload.isExpired())
                    Button("Scan Again") { phase = .scanning }
                } header: {
                    Text("Pair This Phone?")
                } footer: {
                    Text(payload.isExpired() ? "This code has expired. Make a new one on your machine." : "Pair only with a machine you trust: what you make on this phone is sent to it.")
                        .foregroundStyle(.secondaryText)
                }
            case let .pairing(name):
                Label { Text("Pairing with \(name)…") } icon: { ProgressView() }
            case let .failed(reason):
                Section {
                    Label(reason, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                    Button("Scan Again") { phase = .scanning }
                }
            }
            Section("Paste a Pairing Link") {
                TextField("https://utensils.io/mold/pair#…", text: $pasted, axis: .vertical)
                    .lineLimit(1...3)
                    .accessibilityIdentifier("pairing-link-input")
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .font(.body.monospaced())
                if let payload = try? MobilePairingPayload.parse(pasted) {
                    Text("\(payload.mode.title): \(payload.mode.explanation)").foregroundStyle(.secondaryText)
                }
            }
        }
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Pair") { redeem(pasted) }
                    .accessibilityIdentifier("pairing-link-submit")
                    .disabled(pasted.isEmpty || phase != .scanning)
            }
        }
        .permissionAlert($camera.recovery)
        .task { if DataScannerViewController.isSupported { _ = await camera.request() } }
        .onChange(of: scenePhase) { _, phase in if phase == .active { camera.refresh() } }
        .navigationTitle("Scan a Pairing Code")
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder private var scanner: some View {
        if camera.status == .authorized && DataScannerViewController.isSupported && DataScannerViewController.isAvailable {
            QRScanner(active: phase == .scanning) { scan($0) }
                .accessibilityLabel("Camera viewfinder for the pairing code")
        } else {
            VStack {
                EmptyState(
                    title: String(localized: "The camera isn't available"),
                    symbol: "camera",
                    message: DataScannerViewController.isSupported
                        ? PermissionRecovery.camera(camera.status)?.message ?? String(localized: "The camera is unavailable right now. Paste the pairing link below.")
                        : String(localized: "This device can't scan codes. Paste the pairing link below."))
                if let recovery = PermissionRecovery.camera(camera.status) { PermissionSettingsButton(recovery: recovery) }
            }
        }
    }

    private func scan(_ raw: String) {
        guard phase == .scanning else { return }
        do {
            let payload = try MobilePairingPayload.parse(raw)
            phase = Phase.afterScan(payload)
            if payload.mode == .secure { pair(payload) }
        } catch {
            phase = .failed(error.errorDescription ?? error.sentence)
        }
    }

    private func redeem(_ raw: String) {
        guard phase == .scanning else { return }
        let payload: MobilePairingPayload
        do { payload = try MobilePairingPayload.parse(raw) } catch {
            phase = .failed(error.errorDescription ?? error.sentence)
            return
        }
        pair(payload)
    }

    private func pair(_ payload: MobilePairingPayload) {
        phase = .pairing(payload.name)
        Task {
            do {
                try await hosts.pair(payload, claim: stores.claimPairing)
                done()
            } catch {
                phase = .failed(PairingFailure.sentence(error, name: payload.name))
            }
        }
    }
}

/// VisionKit's live scanner, reading QR codes only.
private struct QRScanner: UIViewControllerRepresentable {
    let active: Bool
    let found: (String) -> Void

    func makeUIViewController(context: Context) -> DataScannerViewController {
        let scanner = DataScannerViewController(
            recognizedDataTypes: [.barcode(symbologies: [.qr])],
            qualityLevel: .balanced, isHighlightingEnabled: true)
        scanner.delegate = context.coordinator
        return scanner
    }

    func updateUIViewController(_ scanner: DataScannerViewController, context: Context) {
        context.coordinator.found = found
        if active, !scanner.isScanning { try? scanner.startScanning() }
        if !active, scanner.isScanning { scanner.stopScanning() }
    }

    func makeCoordinator() -> Coordinator { Coordinator(found: found) }

    final class Coordinator: NSObject, DataScannerViewControllerDelegate {
        var found: (String) -> Void
        init(found: @escaping (String) -> Void) { self.found = found }

        func dataScanner(_ scanner: DataScannerViewController, didAdd items: [RecognizedItem],
                         allItems: [RecognizedItem]) {
            for case let .barcode(code) in items {
                if let text = code.payloadStringValue { found(text); return }
            }
        }
    }
}
