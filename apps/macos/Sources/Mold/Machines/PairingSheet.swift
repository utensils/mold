import MoldClient
import SwiftUI

/// One pairing session's QR code, and the countdown until it dies.
///
/// The countdown reads `TimelineView`, not a `Timer`: this sheet is the only
/// thing that needs the tick, and a `TimelineView` simply stops ticking when
/// it closes rather than needing an `invalidate()` somewhere.
struct PairingSheet: View {
    @Environment(PairingStore.self) private var pairing
    @Environment(RemotePairingStore.self) private var remote
    @Environment(\.dismiss) private var dismiss
    let host: MoldHost
    var showsDone = true
    @State private var mode: HostConnectionMode = .secure
    @State private var directAddress = ""
    @State private var codeBaseURL: URL?
    @State private var preparationFailure: String?
    @State private var codeGeneration = 0

    enum SheetState: Equatable {
        case waiting
        case code(MobilePairingPayload)
        case noCode
        /// The machine refused or could not be reached, in words.
        case failed(String)
    }

    /// Pure, so the three branches are tested with no view.
    static func resolve(_ session: PairingSession?, failure: String? = nil, baseURL: URL, name: String) -> SheetState {
        guard let session else { return failure.map(SheetState.failed) ?? .waiting }
        guard let payload = MobilePairingPayload(session: session, baseURL: baseURL, name: name) else {
            return .noCode
        }
        return .code(payload)
    }

    /// Whether opening the sheet must mint a code: yes unless a session for
    /// THIS machine is already in flight and unexpired -- reopening the
    /// sheet inside the two-minute window shows the code a phone may already
    /// be scanning, rather than killing it (M7 UAT: a seeded fixture session
    /// was being replaced by a request the fixture then refused).
    static func needsFreshCode(session: PairingSession?, sessionHost: MoldHost.ID?, host: MoldHost.ID, now: Date, mode: HostConnectionMode = .secure) -> Bool {
        guard let session, sessionHost == host else { return true }
        guard (session.mode ?? .secure) == mode else { return true }
        let expiry = mode == .regular ? session.expiresAt : session.secureInvitation?.invite.expiresAt
        guard let expiresAt = expiry else { return true }
        return Countdown.resolve(expiresAt: expiresAt, now: now) == .expired
    }

    /// Reads the MACHINE's own `expires_at`, never a client-side clock
    /// started when the sheet opened -- `resolve(_:now:)` takes `now`
    /// explicitly so a fixed instant is what a test pins.
    enum Countdown: Equatable {
        case remaining(TimeInterval)
        case expired

        /// `expires_at` is unix SECONDS, not milliseconds: `auth.rs:216`
        /// adds `PAIRING_TOKEN_TTL_SECS` to `unix_timestamp()`, which is
        /// `.as_secs()` (`auth.rs:633-638`), and `routes.rs:9502` puts that
        /// on the wire. Studio subtracts it from `Date.now() / 1000` for the
        /// same reason (`MobilePairingCard.vue:36-37`). `PairedClient
        /// .lastUsedAtMs` IS milliseconds -- that is a different field, and
        /// dividing this one by 1000 put every live code in 1970 and drew
        /// it as "Expired".
        static func resolve(expiresAt: UInt64, now: Date) -> Countdown {
            let remaining = TimeInterval(expiresAt) - now.timeIntervalSince1970
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
        let session = pairing.sessionHost == host.id && (pairing.session?.mode ?? .secure) == mode ? pairing.session : nil
        return Self.resolve(session, failure: preparationFailure ?? pairing.sessionFailure[host.id], baseURL: codeBaseURL ?? host.baseURL, name: host.name)
    }

    var body: some View {
        VStack(spacing: 16) {
            Text("Pair “\(host.name)”").font(.headline)
            Picker("Connection", selection: $mode) {
                ForEach(HostConnectionMode.allCases, id: \.self) { Text($0.title).tag($0) }
            }.pickerStyle(.segmented)
            Text(mode.explanation).font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if mode == .regular && host.id != MoldEngine.localHostID {
                TextField("Direct HTTP or HTTPS address", text: $directAddress)
                    .textFieldStyle(.roundedBorder).onSubmit { Task { await createCode() } }
                Button("Create Regular Code") { Task { await createCode() } }
            }
            if mode == .regular && host.id == MoldEngine.localHostID {
                Text("Regular enables authenticated direct access on this Mac. You can turn it off in Remote Access settings.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            switch state {
            case .waiting:
                ProgressView().frame(width: 220, height: 220)
            case .noCode:
                unavailable
            case let .failed(reason):
                VStack(spacing: 12) {
                    Text(reason)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Try Again") { Task { await createCode() } }
                }
                .frame(width: 220, height: 220)
            case let .code(payload):
                if let url = payload.url {
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        codeBody(payload: payload, url: url, now: context.date)
                    }
                } else {
                    unavailable
                }
            }
            if showsDone {
                HStack {
                    Spacer()
                    Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(24)
        .frame(width: 320)
        .task(id: mode) {
            preparationFailure = nil
            if directAddress.isEmpty && DirectHostAddress.isValid(host.baseURL) { directAddress = host.baseURL.absoluteString }
            guard Self.needsFreshCode(session: pairing.session, sessionHost: pairing.sessionHost,
                                      host: host.id, now: .now, mode: mode) else { return }
            await createCode()
        }
    }

    private func createCode() async {
        preparationFailure = nil
        codeGeneration += 1
        let generation = codeGeneration
        let selectedMode = mode
        do {
            if selectedMode == .regular {
                let address: URL
                if host.id == MoldEngine.localHostID { address = try await remote.prepareRegular(host) }
                else {
                    guard let entered = URL(string: directAddress), DirectHostAddress.isValid(entered) else {
                        preparationFailure = "Enter this machine’s direct HTTP or HTTPS address. Mold proxy always uses Secure."
                        return
                    }
                    address = entered
                }
                try Task.checkCancellation()
                guard generation == codeGeneration, mode == selectedMode else { return }
                codeBaseURL = address
                await pairing.createSession(on: host.id, mode: .regular, directURL: address)
            } else if host.id == MoldEngine.localHostID {
                let ready = try await remote.prepare(host)
                try Task.checkCancellation()
                guard generation == codeGeneration, mode == selectedMode else { return }
                codeBaseURL = ready.baseURL
            } else {
                codeBaseURL = host.baseURL
                await pairing.createSession(on: host.id)
            }
        } catch {
            guard !Task.isCancelled, generation == codeGeneration, mode == selectedMode else { return }
            preparationFailure = error.failureSentence
        }
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
            // The code is a universal link: the Camera app opens Mold Studio
            // on the phone, which asks before it pairs.
            Text("Scan this code on your phone, or copy its link into Add a Machine on another device.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
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
                Button("New Code") { Task { await createCode() } }
            }
        }
    }
}
