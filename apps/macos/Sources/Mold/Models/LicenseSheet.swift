import MoldClient
import SwiftUI

/// What a held licence looks like, rendered from the 403 payload alone --
/// `LicenseRefusal` (`types.rs:12604-12618`) -- and never from prose:
/// the name, the server's own one-sentence summary, a link to the browsable
/// page, and a caption naming exactly the pinned text and hash Accept binds
/// to. `LicenseRefusal` carries no `requiredBy`/`requiredByStyles` -- those
/// ride only on `ThirdPartyLicense` from `GET /api/licenses` -- so unlike a
/// full licence row this sheet never lists the models it gates; that is a
/// later "Show Licence…" surface's job, reading the full listing instead of
/// a held refusal.
struct LicenseSheet: View {
    let pending: DownloadStore.PendingLicense
    @Environment(DownloadStore.self) private var downloads
    @State private var isAccepting = false

    private var refusal: LicenseRefusal { pending.refusal }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            ScrollView {
                Text(refusal.summary)
                    .font(.callout)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 160)
            Link("Read the terms ↗", destination: Self.link(refusal.canonical))
            Text(Self.caption(for: refusal))
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Spacer(minLength: 0)
            actions
        }
        .padding(20)
        .frame(width: 420, height: 340)
    }

    @ViewBuilder private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(refusal.name).font(.title2.weight(.semibold))
            if let sentence = Self.mismatchSentence(for: pending) {
                Label(sentence, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var actions: some View {
        HStack {
            Spacer()
            // `pendingLicense` is `internal(set)` for exactly this: Cancel
            // needs no store method, only to let go of what it is holding.
            Button("Cancel") { downloads.pendingLicense = nil }
                .keyboardShortcut(.cancelAction)
            Button("Accept and Download") { accept() }
                .keyboardShortcut(.defaultAction)
                .disabled(isAccepting)
        }
    }

    private func accept() {
        isAccepting = true
        Task {
            await downloads.accepted(pending)
            isAccepting = false
        }
    }

    /// The server's own reason gets a title; a terms mismatch is this
    /// sheet's OWN warning line, never a second title -- the licence's name
    /// always answers "what am I looking at" (design fact/decision, M5).
    static func mismatchSentence(for pending: DownloadStore.PendingLicense) -> String? {
        pending.mismatch ? "This machine already accepted different terms for this licence." : nil
    }

    /// What Accept actually binds to -- the same `(url, sha256)` pair
    /// `LicenseRefusal.acceptance` sends -- spelled out so nobody accepts
    /// text the button did not show them.
    static func caption(for refusal: LicenseRefusal) -> String {
        "Pinned at \(refusal.url) — sha256 \(refusal.sha256.prefix(12))…"
    }

    private static func link(_ string: String) -> URL {
        URL(string: string) ?? URL(string: "about:blank")!
    }
}
