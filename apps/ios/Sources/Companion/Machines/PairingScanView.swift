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

    enum Phase: Equatable {
        case scanning
        case pairing(String)
        case failed(String)
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
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .font(.body.monospaced())
                Button("Pair") { redeem(pasted) }
                    .disabled(pasted.isEmpty || phase != .scanning)
            }
        }
        .navigationTitle("Scan a Pairing Code")
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder private var scanner: some View {
        if DataScannerViewController.isSupported && DataScannerViewController.isAvailable {
            QRScanner(active: phase == .scanning) { redeem($0) }
                .accessibilityLabel("Camera viewfinder for the pairing code")
        } else {
            EmptyState(
                title: String(localized: "The camera isn't available"),
                symbol: "camera",
                message: DataScannerViewController.isSupported
                    ? String(localized: "Allow camera access for Mold Studio in Settings, or paste the pairing link below.")
                    : String(localized: "This device can't scan codes. Paste the pairing link below."))
        }
    }

    private func redeem(_ raw: String) {
        guard phase == .scanning else { return }
        let payload: MobilePairingPayload
        do { payload = try MobilePairingPayload.parse(raw) } catch {
            phase = .failed(error.errorDescription ?? error.localizedDescription)
            return
        }
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
