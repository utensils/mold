import Foundation

/// HTTP semantics inside an authenticated encrypted connection. An injected implementation
/// also lets contract tests exercise routes without opening a network connection.
public protocol MoldHTTPTransport: Sendable {
    func response(for request: URLRequest, body: MoldRequestBody) async throws
        -> (MoldResponseBytes, HTTPURLResponse)
}

public enum MoldRequestBody: Sendable {
    case data(Data)
    case file(URL)

    var length: Int {
        get throws {
            switch self {
            case let .data(data): data.count
            case let .file(url): try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            }
        }
    }
}

public enum SecureConnectionError: Error, Equatable, LocalizedError, Sendable {
    case pairingRequired, authenticationFailed, protocolViolation, closed, capacity, requestTooLarge
    public var errorDescription: String? {
        switch self {
        case .pairingRequired: "Pair this machine again to establish an encrypted connection."
        case .authenticationFailed: "This connection could not verify the paired machine."
        case .protocolViolation: "The encrypted connection received an invalid response."
        case .closed: "The encrypted connection was interrupted."
        case .capacity: "This machine has too many active requests. Try again shortly."
        case .requestTooLarge: "This request exceeds the machine's 64 MiB upload limit."
        }
    }
}
