import Foundation

/// The biggest request any mold machine will accept.
///
/// `MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024` (`crates/mold-server/src/lib.rs:178`),
/// applied to every route by the body-limit layer (`lib.rs:1347`). It is the
/// practical ceiling on a queue transfer too, since the admission body
/// carries the exported bytes verbatim (`HTTPBackend+Transfer.swift`).
///
/// One constant and ONE sentence, because the sentence is what a person
/// actually reads: the transfer refusal used to say "about 48 MB", a number
/// that matches nothing in this repo.
public enum RequestBodyLimit {
    public static let bytes = 64 * 1024 * 1024

    /// "64 MB" -- `.byteCount`'s own spelling of the limit, so the words and
    /// the number can never drift apart. Binary count style: the server's
    /// limit is 64 MiB and `.file` would call that 67.1 MB.
    public static var sentence: String {
        Int64(bytes).formatted(.byteCount(style: .binary))
    }
}
