import MoldClient
import SwiftUI

/// One pairing session's QR code, and the countdown until it dies.
///
/// The countdown reads `TimelineView`, not a `Timer`: this sheet is the only
/// thing that needs the tick, and a `TimelineView` simply stops ticking when
/// it closes rather than needing an `invalidate()` somewhere.
struct PairingSheet: View {
    @Environment(PairingStore.self) private var pairing
    @Environment(\.dismiss) private var dismiss
    let host: MoldHost

    enum SheetState: Equatable {
        case waiting
        case code(MobilePairingPayload)
        case noCode
    }

    /// Pure, so the three branches are tested with no view.
    static func resolve(_ session: PairingSession?, baseURL: URL, name: String) -> SheetState {
        guard let session else { return .waiting }
        guard let payload = MobilePairingPayload(session: session, baseURL: baseURL, name: name) else {
            return .noCode
        }
        return .code(payload)
    }

    /// Reads the MACHINE's own `expires_at`, never a client-side clock
    /// started when the sheet opened -- `resolve(_:now:)` takes `now`
    /// explicitly so a fixed instant is what a test pins.
    enum Countdown: Equatable {
        case remaining(TimeInterval)
        case expired

        static func resolve(expiresAt: UInt64, now: Date) -> Countdown {
            let remaining = TimeInterval(expiresAt) / 1_000 - now.timeIntervalSince1970
            return remaining > 0 ? .remaining(remaining) : .expired
        }

        var sentence: String {
            switch self {
            case .expired:
                return "Expired"
            case let .remaining(seconds):
                let formatted = Duration.seconds(max(0, seconds))
                    .formatted(.units(allowed: [.minutes, .seconds], width: .narrow))
                return "Expires in \(formatted)"
            }
        }
    }

    /// The store's session belongs to whichever host started it last -- this
    /// sheet draws only its own, so switching machines mid-flow shows
    /// "waiting" rather than a code for somewhere else.
    private var state: SheetState {
        let session = pairing.sessionHost == host.id ? pairing.session : nil
        return Self.resolve(session, baseURL: host.baseURL, name: host.name)
    }

    var body: some View {
        VStack(spacing: 16) {
            Text("Pair “\(host.name)”").font(.headline)
            switch state {
            case .waiting:
                ProgressView().frame(width: 220, height: 220)
            case .noCode:
                unavailable
            case let .code(payload):
                if let url = payload.url {
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        codeBody(payload: payload, url: url, now: context.date)
                    }
                } else {
                    unavailable
                }
            }
            HStack {
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 320)
        .task { await pairing.createSession(on: host.id) }
    }

    private var unavailable: some View {
        Text("This machine has no code to show.")
            .foregroundStyle(.secondary)
            .frame(width: 220, height: 220)
    }

    @ViewBuilder private func codeBody(payload: MobilePairingPayload, url: URL, now: Date) -> some View {
        let countdown = payload.expiresAt.map { Countdown.resolve(expiresAt: $0, now: now) }
        let expired = countdown == .expired
        VStack(spacing: 12) {
            QRCodeImage(payload: url.absoluteString)
                .frame(width: 220, height: 220)
                .opacity(expired ? 0.3 : 1)
                .accessibilityLabel("Pairing code for \(host.name), \(countdown?.sentence ?? "no expiry set")")
            HStack(spacing: 6) {
                Text(url.absoluteString).font(.caption).monospaced()
                    .textSelection(.enabled).lineLimit(1)
                CopyButton(what: "Code", value: url.absoluteString)
            }
            Text(countdown?.sentence ?? "This code does not expire")
                .font(.caption)
                .foregroundStyle(.secondary)
            if expired {
                Button("New Code") { Task { await pairing.createSession(on: host.id) } }
            }
        }
    }
}
