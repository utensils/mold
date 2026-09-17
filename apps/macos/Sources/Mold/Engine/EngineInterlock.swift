import Foundation

/// The check that stops a second engine being started on one mold home.
///
/// `MoldEngine.start()` picked a free port and started `run_server` with no
/// check at all, so with Mold Desktop (or `mold serve`) already running, one
/// Mac ended up with two `run_server` processes on one home. The gallery
/// writer lease is shared by design; queue OWNERSHIP is not — two owner
/// records under `$MOLD_HOME/queue-owners/` means "adopt none, mint fresh,
/// report each orphan", and the other app's queued work is stranded
/// (review 05-M6).
///
/// The probe is the Tauri app's (`desktop/src-tauri/src/commands.rs:283-311`):
/// ask the well-known address, and refuse by NAME rather than by guess. It
/// deliberately does not read the writer lease — the lease lives under the
/// output directory, which is `None` on a default config, so its absence
/// would prove nothing.
enum EngineInterlock {
    /// Where every other mold on this Mac binds unless told otherwise: the
    /// CLI's default `MOLD_HOST`, the Tauri app's sidecar, `mold serve`.
    static let wellKnown = URL(string: "http://127.0.0.1:7680")!

    static func refusal(at address: URL) -> String {
        "A mold server is already answering at \(address.absoluteString). Two engines sharing "
            + "one mold home strand each other's queued work, so Mold won't start a second — "
            + "quit Mold Desktop or stop `mold serve`, then try again."
    }

    /// The sentence to refuse with, or `nil` when nothing else answers.
    static func otherServer(
        probe: (URL) async -> Bool = EngineInterlock.answers
    ) async -> String? {
        await probe(wellKnown) ? refusal(at: wellKnown) : nil
    }

    /// Any HTTP reply counts, including a 401: a keyed mold refusing us is
    /// still a mold holding the address.
    private static func answers(_ address: URL) async -> Bool {
        var request = URLRequest(url: address.appending(path: "api/status"))
        request.timeoutInterval = 2
        guard let (_, response) = try? await URLSession.shared.data(for: request) else {
            return false
        }
        return response is HTTPURLResponse
    }
}
