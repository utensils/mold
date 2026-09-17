import Foundation

/// What a `URLError` means to this app.
enum TransportFailure {
    /// A cancelled request is the app changing its mind -- a `.task(id:)`
    /// re-keying, a view going away -- not the machine failing, so it must
    /// never present as `.unreachable`. Every other `URLError` still becomes
    /// the same reachability failure as before.
    static func from(_ error: URLError) -> Error {
        error.code == .cancelled ? CancellationError() : MoldClientError.unreachable(error.localizedDescription)
    }
}
