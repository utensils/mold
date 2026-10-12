import Foundation
import Network
import Testing
@testable import MoldClient

struct SecureAdmissionRetryTests {
    @Test(arguments: [429, 503, 401]) func actualUpgradeRefusalHasBoundedFreshSockets(status: Int) async throws {
        let server = try UpgradeRefusal(status: status)
        let endpoint = try await server.start()
        defer { server.listener.cancel() }
        let owner = try SecureIdentity(), device = try SecureIdentity()
        let binding = try owner.binding(), peer = try device.binding()
        let namespace = try SecureAdmission.namespace(binding)
        let expiry = UInt64(Date().timeIntervalSince1970) + 60
        let id = String(repeating: "a", count: 32)
        var bytes = Data("MOLD3-GRANT\0".utf8)
        bytes.append(try SecureHex.decode(namespace, count: 16))
        bytes.append(try SecureHex.decode(peer.signingKey, count: 32))
        bytes.append(try SecureHex.decode(peer.noiseKey, count: 32))
        bytes.append(try SecureHex.decode(id, count: 16))
        bytes.append(SecureHex.u64(expiry)); bytes.append(SecureHex.u64(0))
        let host = SecureHost(binding: binding, namespace: namespace, websocketURL: URL(string: "wss://proxy.example")!,
                              grant: .init(id: id, expiresAt: expiry, epoch: 0, signature: SecureHex.encode(try owner.sign(bytes))))
        do {
            _ = try await SecureWebSocketConnection.connect(host: host, identity: device, endpoint: endpoint, timeout: 4)
            Issue.record("Refused upgrade must not establish a connection")
        } catch {
            #expect((error is SecureAdmissionRetry.ProxyBusy) == (status != 401))
        }
        #expect(server.attempts == (status == 401 ? 1 : 3))
    }
    @Test func onlyBusyUpgradeBeforeSocketOpenIsRetryable() throws {
        let url = URL(string: "https://proxy.example/production?v=3")!
        for status in [429, 503] {
            let response = try #require(HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: nil))
            #expect(SecureAdmissionRetry.isBusyUpgrade(response: response, opened: false))
            #expect(!SecureAdmissionRetry.isBusyUpgrade(response: response, opened: true))
        }
        for status in [101, 200, 301, 400, 401, 403, 404, 500] {
            let response = try #require(HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: nil))
            #expect(!SecureAdmissionRetry.isBusyUpgrade(response: response, opened: false))
        }
        #expect(!SecureAdmissionRetry.isBusyUpgrade(response: nil, opened: false))
    }
    private actor Attempts {
        var count = 0
        func next() -> Int { count += 1; return count }
    }

    @Test func retriesOnlyPreAdmissionBusyAndCapsAtThreeFreshAttempts() async throws {
        let attempts = Attempts()
        let result = try await SecureAdmissionRetry.open(timeout: 2) { _ in
            let count = await attempts.next()
            if count < 3 { throw SecureAdmissionRetry.ProxyBusy() }
            return count
        }
        #expect(result == 3)
        let exhausted = Attempts()
        await #expect(throws: SecureAdmissionRetry.ProxyBusy.self) {
            try await SecureAdmissionRetry.open(timeout: 2) { _ -> Int in
                _ = await exhausted.next()
                throw SecureAdmissionRetry.ProxyBusy()
            }
        }
        #expect(await exhausted.count == 3)
    }

    @Test func authenticationProtocolAndDispatchedFailuresNeverRetry() async {
        for failure in [SecureConnectionError.authenticationFailed, .protocolViolation, .closed] {
            let attempts = Attempts()
            await #expect(throws: failure) {
                try await SecureAdmissionRetry.open(timeout: 2) { _ -> Int in
                    _ = await attempts.next()
                    throw failure
                }
            }
            #expect(await attempts.count == 1)
        }
    }

    @Test func cancellationAndExistingDeadlineBoundBackoff() async throws {
        let attempts = Attempts()
        let task = Task {
            try await SecureAdmissionRetry.open(timeout: 2) { _ -> Int in
                _ = await attempts.next()
                throw SecureAdmissionRetry.ProxyBusy()
            }
        }
        while await attempts.count == 0 { await Task.yield() }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await attempts.count == 1)
        let short = Attempts()
        await #expect(throws: URLError.self) {
            try await SecureAdmissionRetry.open(timeout: 0.01) { _ -> Int in
                _ = await short.next()
                throw SecureAdmissionRetry.ProxyBusy()
            }
        }
        #expect(await short.count == 1)
    }
}

private final class UpgradeRefusal: @unchecked Sendable {
    let listener: NWListener
    private let queue = DispatchQueue(label: "mold.upgrade-refusal")
    private let lock = NSLock()
    private var count = 0
    var attempts: Int { lock.withLock { count } }
    init(status: Int) throws {
        listener = try NWListener(using: .tcp, on: .any)
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { connection.cancel(); return }
            self.lock.withLock { self.count += 1 }
            connection.start(queue: self.queue)
            connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { _, _, _, _ in
                let response = Data("HTTP/1.1 \(status) Refused\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".utf8)
                connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
            }
        }
    }
    func start() async throws -> URL {
        listener.start(queue: queue)
        for _ in 0..<100 {
            if let port = listener.port, port.rawValue > 0 { return URL(string: "ws://127.0.0.1:\(port.rawValue)")! }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw URLError(.timedOut)
    }
}
