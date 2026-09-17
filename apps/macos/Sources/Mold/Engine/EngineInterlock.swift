import Foundation

/// Whether something else is already publishing into this mold home.
///
/// The first version asked `http://127.0.0.1:7680/api/status`, which detects
/// nothing: this app always binds an EPHEMERAL port and Mold Desktop only
/// PREFERS 7680 (`desktop/src-tauri/src/server.rs:119-131`), so the ordinary
/// sequence — native engine first, Desktop second — saw neither side. The
/// authority mold actually has is the gallery writer lease, held SHARED for
/// the life of every writing process with its pid in the body, and
/// `mold_engine_home_writer_pid` reads it without writing anything
/// (review F3).
///
/// It is an ADVISORY and not a refusal, which is a change of policy from
/// review 05-M6, decided from the code rather than from the finding's wording:
///
/// * `queue_journal.rs:203-213` designs for it — "two servers sharing one
///   `MOLD_HOME` differ only by port" — and identifies a returning server by a
///   recorded instance hint rather than by exclusion.
/// * A queue this server does not adopt is REPORTED as an orphan at startup
///   ("Retained queues this server did not take. Reported at startup so work is
///   never silently stranded"), so the harm 05-M6 names is visible, not silent.
/// * CLAUDE.md is explicit that the writer lease is shared: "two servers still
///   share a home".
///
/// Refusing would therefore stop a legitimate setup — the native app beside a
/// `mold serve` on one home — that mold itself supports. What was missing is
/// that nobody was TOLD, and that is what this fixes.
enum EngineInterlock {
    enum HomeWriter: Equatable {
        /// Nobody is publishing here, or the lease is a dead process's
        /// leftover, which every reader already treats as stale.
        case none
        /// A live writer, named where its lease body could be read.
        case live(pid: Int64?)
        /// Could not be determined. Never an obstacle: absence of an answer is
        /// not evidence.
        case unknown
    }

    static func advisory(for writer: HomeWriter) -> String? {
        switch writer {
        case .none, .unknown:
            nil
        case let .live(pid):
            "Another mold is already publishing into this mold home"
                + (pid.map { " (pid \($0))" } ?? "")
                + ". They will share the library but keep separate queues, and each will report "
                + "the other's unfinished work as orphaned rather than run it."
        }
    }

    /// Asked once, just before starting — never while this app's own engine
    /// holds the lease, which would name us.
    static func homeWriter(probe: () -> Int64 = EngineInterlock.writerPID) -> HomeWriter {
        switch probe() {
        case 0: .none
        case let pid where pid > 0: .live(pid: pid)
        case -2: .live(pid: nil)
        default: .unknown
        }
    }

    /// `nonisolated` so it can also be the default argument above: a plain C
    /// call that touches no actor state.
    private nonisolated static func writerPID() -> Int64 {
        #if MOLD_EMBEDDED_ENGINE
        mold_engine_home_writer_pid()
        #else
        -1
        #endif
    }
}
