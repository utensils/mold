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

    /// Anything the connection itself threw. A Secure machine's requests ride
    /// a WebSocket, whose dropped socket throws a POSIX or network `NSError`
    /// rather than a `URLError`; left alone it read as "That action failed"
    /// in a banner instead of the machine being offline. Replies, refusals
    /// and the app's own errors pass through untouched.
    static func from(any error: any Error) -> Error {
        if let error = error as? URLError { return from(error) }
        if error is MoldClientError || error is CancellationError { return error }
        if case SecureConnectionError.closed = error { return MoldClientError.unreachable(error.localizedDescription) }
        let system = error as NSError
        guard connectionDomains.contains(system.domain) else { return error }
        return MoldClientError.unreachable(system.localizedDescription)
    }

    static let connectionDomains: Set<String> = [NSPOSIXErrorDomain, "kCFErrorDomainCFNetwork", "Network.NWError", "NWErrorDomain"]
}
