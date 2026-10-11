import Foundation
import Network

/// Owns the socket, deadline and continuation together. A stale Bonjour
/// advertisement must never leave Add spinning or retain a background socket.
@MainActor
final class NearbyAddressResolution {
    private let connection: NWConnection
    private var continuation: CheckedContinuation<String, any Error>?
    private var deadline: Task<Void, Never>?
    private var finished = false

    init(endpoint: NWEndpoint) { connection = NWConnection(to: endpoint, using: .tcp) }

    func address(timeout: Duration) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            guard !finished else { return continuation.resume(throwing: CancellationError()) }
            self.continuation = continuation
            connection.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in self?.changed(state) }
            }
            deadline = Task { [weak self] in
                do { try await Task.sleep(for: timeout) } catch { return }
                self?.finish(.failure(URLError(.timedOut)))
            }
            connection.start(queue: .main)
        }
    }

    func cancel() { finish(.failure(CancellationError())) }

    private func changed(_ state: NWConnection.State) {
        switch state {
        case .ready:
            if case let .hostPort(host, port)? = connection.currentPath?.remoteEndpoint {
                finish(.success(NearbyBrowser.address(host: host, port: port)))
            } else { finish(.failure(URLError(.cannotFindHost))) }
        case let .failed(error), let .waiting(error): finish(.failure(error))
        case .cancelled: cancel()
        default: break
        }
    }

    private func finish(_ result: Result<String, any Error>) {
        guard !finished else { return }
        finished = true
        deadline?.cancel()
        deadline = nil
        connection.stateUpdateHandler = nil
        connection.cancel()
        let waiting = continuation
        continuation = nil
        waiting?.resume(with: result)
    }
}
