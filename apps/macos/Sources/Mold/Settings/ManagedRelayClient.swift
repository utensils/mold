import Foundation
import MoldClient

/// Talks only to the central enrollment service; never sends a Mold API key.
struct ManagedRelayClient: Sendable {
    let origin: URL
    let session: URLSession

    init(origin: URL = ManagedRelayEnrollment.controlOrigin, session: URLSession = Self.enrollmentSession) {
        self.origin = origin
        self.session = session
    }

    static let enrollmentSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.urlCache = nil
        configuration.timeoutIntervalForResource = 12
        return URLSession(configuration: configuration)
    }()

    func enroll(previous: ManagedRelayEnrollment?) async throws -> ManagedRelayEnrollment {
        if let previous {
            try previous.validate(control: origin, requiresLiveLease: false)
            let (data, status) = try await request(method: "POST", owner: previous)
            // An expired/revoked namespace may no longer exist. Replace it only
            // after a definitive refusal, never after an uncertain timeout.
            if ![401, 403, 404, 410].contains(status) {
                return try enrollment(data: data, status: status)
            }
        }
        let (data, status) = try await request(method: "POST", owner: nil)
        return try enrollment(data: data, status: status)
    }

    func revoke(_ owner: ManagedRelayEnrollment) async throws {
        try owner.validate(control: origin, requiresLiveLease: false)
        let (_, status) = try await request(method: "DELETE", owner: owner)
        guard [200, 204, 401, 404, 410].contains(status) else { throw ManagedRelayFailure.unavailable }
    }

    private func enrollment(data: Data, status: Int) throws -> ManagedRelayEnrollment {
        guard status == 200 || status == 201 else {
            throw status == 429 ? ManagedRelayFailure.capacity : .unavailable
        }
        let record = try MoldJSON.decoder.decode(ManagedRelayEnrollment.self, from: data)
        try record.validate(control: origin)
        return record
    }

    private func request(method: String, owner: ManagedRelayEnrollment?) async throws -> (Data, Int) {
        guard origin == ManagedRelayEnrollment.controlOrigin else { throw ManagedRelayFailure.invalidEnrollment }
        var url = origin.appending(path: "_mold/relay/enroll")
        if let owner { url.append(path: owner.hostId) }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 12
        request.httpShouldHandleCookies = false
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let owner { request.setValue("Bearer \(owner.token)", forHTTPHeaderField: "Authorization") }
        let (bytes, response) = try await session.bytes(for: request, delegate: ManagedRelayNoRedirect())
        defer { bytes.task.cancel() }
        guard let response = response as? HTTPURLResponse else { throw ManagedRelayFailure.unavailable }
        var data = Data()
        for try await byte in bytes {
            guard data.count < 16_384 else { throw ManagedRelayFailure.invalidEnrollment }
            data.append(byte)
        }
        return (data, response.statusCode)
    }
}

private final class ManagedRelayNoRedirect: NSObject, URLSessionTaskDelegate, Sendable {
    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                               willPerformHTTPRedirection response: HTTPURLResponse,
                               newRequest request: URLRequest,
                               completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
