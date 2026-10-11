import Foundation
import MoldClient

/// Discovers public relay configuration; identity and admission stay in the engine.
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

    func discover(namespace: String) async throws -> ManagedRelayEnrollment {
        guard origin == ManagedRelayEnrollment.controlOrigin else { throw ManagedRelayFailure.invalidEnrollment }
        var request = URLRequest(url: origin.appending(path: "_mold/relay/info"))
        request.timeoutInterval = 12
        request.httpShouldHandleCookies = false
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (bytes, response) = try await session.bytes(for: request, delegate: ManagedRelayNoRedirect())
        defer { bytes.task.cancel() }
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else { throw ManagedRelayFailure.unavailable }
        var data = Data()
        for try await byte in bytes {
            guard data.count < 16_384 else { throw ManagedRelayFailure.invalidEnrollment }
            data.append(byte)
        }
        let metadata = try MoldJSON.decoder.decode(Metadata.self, from: data)
        guard metadata.protocolVersion == 3, metadata.hostDomain == origin.host,
              let publicURL = URL(string: "https://\(namespace).\(metadata.hostDomain)") else { throw ManagedRelayFailure.invalidEnrollment }
        let record = ManagedRelayEnrollment(hostId: namespace, publicUrl: publicURL, relayUrl: metadata.websocketUrl)
        try record.validate(control: origin)
        return record
    }

    private struct Metadata: Decodable {
        let protocolVersion: Int
        let websocketUrl: URL
        let hostDomain: String
        enum CodingKeys: String, CodingKey { case protocolVersion = "protocol", websocketUrl, hostDomain }
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
