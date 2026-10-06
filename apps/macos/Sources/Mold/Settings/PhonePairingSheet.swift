import MoldClient
import SwiftUI

/// Presented immediately; setup and retry do not change the Settings layout.
struct PhonePairingSheet: View {
    @Environment(RemotePairingStore.self) private var remote
    @Environment(\.dismiss) private var dismiss
    let host: MoldHost
    @State private var resolvedHost: MoldHost?
    @State private var failure: String?
    @State private var attempt = 0

    var body: some View {
        VStack(spacing: 16) {
            if let resolvedHost {
                PairingSheet(host: resolvedHost, showsDone: false)
            } else {
                Text("Pair your phone").font(.headline)
                if let failure {
                    Text(failure).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Try Again") { self.failure = nil; attempt += 1 }
                } else {
                    ProgressView("Connecting This Mac to Mold proxy…")
                        .padding(.vertical, 36)
                }
            }
            HStack {
                Spacer()
                Button(resolvedHost == nil ? "Close" : "Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 352)
        .onChange(of: remote.state) {
            guard host.id == MoldEngine.localHostID else { return }
            if case let .failed(reason) = remote.state { resolvedHost = nil; failure = reason }
            if remote.state == .off { resolvedHost = nil; failure = "Remote access is stopped. Try pairing again." }
        }
        .task(id: attempt) {
            do {
                if host.id == MoldEngine.localHostID {
                    let ready = try await remote.prepare(host)
                    try Task.checkCancellation()
                    resolvedHost = ready
                } else if RemoteAccessSettings.canPair(host) {
                    resolvedHost = host
                } else {
                    throw ManagedRelayFailure.engineNotRunning
                }
            } catch {
                guard !Task.isCancelled else { return }
                failure = error.failureSentence
            }
        }
    }
}
