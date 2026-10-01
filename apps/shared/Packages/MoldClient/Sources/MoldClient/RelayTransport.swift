import CryptoKit
import Foundation

/// Additive HTTP adaptation for the Lambda facade; direct hosts retain their wire.
enum RelayTransport {
    struct Info: Decodable, Sendable {
        let `protocol`: Int
        let uploadThreshold: Int
        let maxBodyBytes: Int
        let objectOrigin: String?
        static let direct = Info(protocol: 0, uploadThreshold: 0, maxBodyBytes: 0, objectOrigin: nil)
    }
    struct Grant: Decodable {
        let id: String
        let url: String
        let headers: [String: String]
        let expiresAt: UInt64
    }
    struct Object: Decodable {
        let url: String
        let status: Int
        let headers: [String: String]
    }
    static let cache = RelayDiscovery()
    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    static func fileSHA256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty { hash.update(data: chunk) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func filePrefix(_ file: URL, limit: Int = 1024 * 1024) throws -> Data {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        return try handle.read(upToCount: limit) ?? Data()
    }
    static func s3Identity(_ url: URL) -> String? {
        guard url.scheme == "https", url.port == nil || url.port == 443, url.user == nil, url.password == nil,
            url.fragment == nil, let host = url.host
        else { return nil }
        let pattern =
            #"^mold-relay-(\d{12})-([a-z]{2}(?:-gov)?-[a-z]+-\d)\.s3\.(?:dualstack\.)?([a-z]{2}(?:-gov)?-[a-z]+-\d)\.amazonaws\.com$"#
        let regex = try! NSRegularExpression(pattern: pattern)
        guard let match = regex.firstMatch(in: host, range: NSRange(host.startIndex..., in: host)) else {
            return nil
        }
        let account = String(host[Range(match.range(at: 1), in: host)!])
        let region = String(host[Range(match.range(at: 2), in: host)!])
        guard region == String(host[Range(match.range(at: 3), in: host)!]) else { return nil }
        return account + ":" + region
    }
    static func objectOrigin(_ value: String) throws -> URL {
        guard let url = URL(string: value), s3Identity(url) != nil, url.path.isEmpty || url.path == "/",
            url.query == nil
        else { throw MoldClientError.malformedResponse }
        return url
    }
    static func objectURL(_ value: String, origin: URL, objectOrigin: String? = nil) throws -> URL {
        guard let url = URL(string: value), url.scheme == "https", url.port == nil || url.port == 443,
            url.user == nil, url.password == nil, url.fragment == nil, url.path.hasPrefix("/_mold/objects/")
        else { throw MoldClientError.malformedResponse }
        if !HostAddress.sameOrigin(url, origin) {
            guard let expected = objectOrigin, let identity = s3Identity(url),
                identity == s3Identity(try Self.objectOrigin(expected))
            else { throw MoldClientError.malformedResponse }
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
            func one(_ name: String) -> String? {
                let values = items.filter { $0.name == name }
                return values.count == 1 ? values.first?.value : nil
            }
            guard one("X-Amz-Algorithm") == "AWS4-HMAC-SHA256", let signature = one("X-Amz-Signature"),
                signature.range(of: #"^[a-fA-F0-9]{64}$"#, options: .regularExpression) != nil,
                let expiry = one("X-Amz-Expires"),
                expiry.range(of: #"^[0-9]+$"#, options: .regularExpression) != nil,
                let seconds = Int(expiry), (1...900).contains(seconds)
            else { throw MoldClientError.malformedResponse }
        }
        return url
    }
    static func uploadURL(_ value: String) throws -> URL {
        guard let url = URL(string: value), s3Identity(url) != nil else { throw MoldClientError.malformedResponse }
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        func one(_ name: String) -> String? {
            let values = items.filter { $0.name == name }
            return values.count == 1 ? values.first?.value : nil
        }
        guard one("X-Amz-Algorithm") == "AWS4-HMAC-SHA256", let signature = one("X-Amz-Signature"),
            signature.range(of: #"^[a-fA-F0-9]{64}$"#, options: .regularExpression) != nil,
            let expiry = one("X-Amz-Expires"), expiry.range(of: #"^[0-9]+$"#, options: .regularExpression) != nil,
            let seconds = Int(expiry), (1...900).contains(seconds)
        else { throw MoldClientError.malformedResponse }
        return url
    }
}

extension RelayTransport {
    static func stagedHeaders(_ headers: [String: String]) -> [String: String] {
        var excluded: Set<String> = [
            "authorization", "x-api-key", "cookie", "host", "content-length",
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailer", "transfer-encoding", "upgrade", "x-amz-content-sha256",
            "x-mold-request-target",
        ]
        for (_, value) in headers.filter({ $0.key.lowercased() == "connection" }) {
            excluded.formUnion(value.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces).lowercased()
            })
        }
        return headers.filter { name, _ in
            let name = name.lowercased()
            return !excluded.contains(name) && !name.hasPrefix("x-mold-viewer-")
                && !name.hasPrefix("x-mold-relay-")
        }
    }
}

actor RelayDiscovery {
    private var origins: [String: RelayTransport.Info] = [:]
    func info(origin: URL, session: URLSession, originalHeaders: [String: String] = [:]) async throws -> RelayTransport.Info {
        let key = origin.scheme! + "://" + origin.host! + ":" + String(origin.port ?? 443)
        if let cached = origins[key] { return cached }
        let url = URL(string: "/_mold/relay/info", relativeTo: origin)!.absoluteURL
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        request.httpShouldHandleCookies = false
        for (name, value) in originalHeaders where ["x-api-key", "authorization"].contains(name.lowercased()) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        let (stream, response) = try await session.bytes(for: request, delegate: RelayNoRedirect())
        defer { stream.task.cancel() }
        guard let http = response as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        if http.statusCode == 404 {
            origins[key] = .direct
            return .direct
        }
        let ceiling = 65_536
        guard http.expectedContentLength <= ceiling else {
            throw MoldClientError.malformedResponse
        }
        let data = try await stream.collected(upTo: ceiling)
        try HTTPRefusal.check(http, data)
        let contentType = http.value(forHTTPHeaderField: "Content-Type")?.split(separator: ";").first?
            .trimmingCharacters(in: .whitespaces).lowercased()
        if http.statusCode == 200, contentType == "text/html",
            http.value(forHTTPHeaderField: "x-mold-relay-protocol") == nil,
            let shell = String(data: data, encoding: .utf8),
            ((shell.contains("<title>mold — studio</title>") && shell.contains("<div id=\"app\"></div>")) ||
                (shell.contains("<title>mold</title>") && shell.contains("<h1>mold is running</h1>")
                    && shell.contains("This binary was built without the web gallery UI bundled.")))
        {
            origins[key] = .direct
            return .direct
        }
        let info = try MoldJSON.decoder.decode(RelayTransport.Info.self, from: data)
        guard info.protocol == 2, info.uploadThreshold > 0,
            info.maxBodyBytes >= info.uploadThreshold, info.maxBodyBytes <= 67_108_864
        else { throw MoldClientError.malformedResponse }
        if let objectOrigin = info.objectOrigin { _ = try RelayTransport.objectOrigin(objectOrigin) }
        origins[key] = info
        return info
    }
}

final class RelayNoRedirect: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest
    ) async -> URLRequest? { nil }
}


final class RelaySameOriginRedirect: NSObject, URLSessionTaskDelegate, Sendable {
    let origin: URL
    init(origin: URL) { self.origin = origin }
    func redirected(_ request: URLRequest) -> URLRequest? {
        guard let url = request.url, HostAddress.sameOrigin(origin, url) else { return nil }
        var request = request
        if request.value(forHTTPHeaderField: "x-mold-request-target") != nil {
            request.setValue(url.path(percentEncoded: true) + (url.query(percentEncoded: true).map { "?" + $0 } ?? ""), forHTTPHeaderField: "x-mold-request-target")
        }
        return request
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest) async -> URLRequest? {
        redirected(request)
    }
}
