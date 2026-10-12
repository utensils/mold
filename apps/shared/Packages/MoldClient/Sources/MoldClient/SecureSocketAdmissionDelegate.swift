import Foundation

/// Only the HTTP upgrade may classify a busy response. Once opened, every
/// subsequent network/authentication failure follows the normal fail-fast path.
final class SecureSocketAdmissionDelegate: NSObject, URLSessionWebSocketDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var didOpen = false
    var opened: Bool { lock.withLock { didOpen } }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) { lock.withLock { didOpen = true } }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest) async -> URLRequest? { nil }
}
