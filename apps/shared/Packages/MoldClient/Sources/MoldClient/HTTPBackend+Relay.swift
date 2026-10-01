import Foundation

extension HTTPBackend {
    /// Bounded consumers with their own URL cache use the same relay adapter as ordinary API calls.
    public static func responseBytes(for request: URLRequest, host: MoldHost, session: URLSession)
        async throws -> (URLSession.AsyncBytes, HTTPURLResponse)
    {
        try await HTTPBackend(host: host, session: session).relayBytes(request)
    }

    func relayDelegate(_ request: URLRequest) -> any URLSessionTaskDelegate {
        if request.url?.scheme == "https", request.value(forHTTPHeaderField: "X-Api-Key") != nil {
            return RelayNoRedirect()
        }
        return redirectGuard
    }

    func relayControl(_ path: String, body: [String: Any], original: URLRequest) throws -> URLRequest {
        var control = request(path)
        control.httpMethod = "POST"
        control.timeoutInterval = original.timeoutInterval
        control.setValue(original.value(forHTTPHeaderField: "X-Api-Key"), forHTTPHeaderField: "X-Api-Key")
        control.setValue("application/json", forHTTPHeaderField: "Content-Type")
        control.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        control.setValue(RelayTransport.sha256(control.httpBody!), forHTTPHeaderField: "x-amz-content-sha256")
        return control
    }

    /// A file is retained as a file through hashing and S3 upload; only the small commit envelope is buffered.
    func relayPrepared(_ original: URLRequest, file: URL? = nil) async throws -> (URLRequest, URL?) {
        guard let url = original.url, url.scheme == "https", HostAddress.sameOrigin(url, host.baseURL),
            url.path.hasPrefix("/api/") || url.path.hasPrefix("/_mold/relay/")
        else { return (original, file) }
        var prepared = original
        if url.path.hasPrefix("/api/") {
            prepared.setValue(
                url.path(percentEncoded: true) + (url.query(percentEncoded: true).map { "?" + $0 } ?? ""),
                forHTTPHeaderField: "x-mold-request-target")
        }
        guard ["POST", "PUT", "PATCH", "DELETE"].contains(original.httpMethod ?? "GET") else { return (prepared, file) }
        let size: Int
        let digest: String
        if let file {
            size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            digest = try RelayTransport.fileSHA256(file)
        } else {
            let data = original.httpBody ?? Data()
            size = data.count
            digest = RelayTransport.sha256(data)
        }
        prepared.setValue(digest, forHTTPHeaderField: "x-amz-content-sha256")
        guard size > 2_097_152 else { return (prepared, file) }
        let info = try await RelayTransport.cache.info(origin: host.baseURL, session: session)
        guard info.protocol == 2 else { return (prepared, file) }
        guard size <= info.maxBodyBytes else {
            throw ResponseCeiling.Exceeded(
                bytes: size, ceiling: info.maxBodyBytes, what: "relay request body")
        }
        let grantRequest = try relayControl(
            "/_mold/relay/uploads",
            body: [
                "method": original.httpMethod ?? "POST",
                "path": url.path(percentEncoded: true)
                    + (url.query(percentEncoded: true).map { "?" + $0 } ?? ""),
                "headers": RelayTransport.stagedHeaders(prepared.allHTTPHeaderFields ?? [:]), "size": size, "sha256": digest,
            ], original: original)
        let (grantData, grantResponse) = try await session.data(
            for: grantRequest, delegate: RelayNoRedirect())
        guard let grantHTTP = grantResponse as? HTTPURLResponse else {
            throw MoldClientError.malformedResponse
        }
        try HTTPRefusal.check(grantHTTP, grantData)
        let grant = try MoldJSON.decoder.decode(RelayTransport.Grant.self, from: grantData)
        guard !grant.id.isEmpty, grant.expiresAt > UInt64(Date().timeIntervalSince1970),
            !grant.headers.keys.contains(where: {
                ["x-api-key", "authorization"].contains($0.lowercased())
                    || $0.lowercased().hasPrefix("x-mold-")
            })
        else { throw MoldClientError.malformedResponse }
        var upload = URLRequest(url: try RelayTransport.uploadURL(grant.url))
        guard let objectOrigin = info.objectOrigin,
            RelayTransport.s3Identity(upload.url!)
                == RelayTransport.s3Identity(try RelayTransport.objectOrigin(objectOrigin))
        else { throw MoldClientError.malformedResponse }
        upload.httpMethod = "PUT"
        upload.timeoutInterval = max(300, original.timeoutInterval)
        upload.allHTTPHeaderFields = grant.headers
        let answer: (Data, URLResponse)
        if let file {
            answer = try await session.upload(for: upload, fromFile: file, delegate: RelayNoRedirect())
        } else {
            answer = try await session.upload(
                for: upload, from: original.httpBody ?? Data(), delegate: RelayNoRedirect())
        }
        guard let uploaded = answer.1 as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        try HTTPRefusal.check(uploaded, answer.0)
        return (try relayControl("/_mold/relay/request", body: ["id": grant.id], original: original), nil)
    }

    func relayObject(_ data: Data, response: HTTPURLResponse, original: URLRequest) async throws -> (
        Data, HTTPURLResponse
    ) {
        guard response.value(forHTTPHeaderField: "x-mold-relay-object") == "1" else {
            return (data, response)
        }
        let object = try MoldJSON.decoder.decode(RelayTransport.Object.self, from: data)
        let url = try await relayObjectURL(object.url)
        var download = URLRequest(url: url)
        download.timeoutInterval = original.timeoutInterval
        download.httpMethod = original.httpMethod == "HEAD" ? "HEAD" : "GET"
        let (bytes, answer) = try await session.data(for: download, delegate: RelayNoRedirect())
        guard let http = answer as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        try HTTPRefusal.check(http, bytes)
        guard (200...599).contains(object.status),
            let restored = HTTPURLResponse(
                url: original.url!, statusCode: object.status,
                httpVersion: "HTTP/1.1", headerFields: object.headers)
        else { throw MoldClientError.malformedResponse }
        return (bytes, restored)
    }

    func relayDownload(_ original: URLRequest) async throws -> (URL, HTTPURLResponse) {
        let (prepared, _) = try await relayPrepared(original)
        var (file, answer) = try await session.download(for: prepared, delegate: relayDelegate(prepared))
        guard var http = answer as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        if http.value(forHTTPHeaderField: "x-mold-relay-object") == "1" {
            let envelopeFile = file
            defer { try? FileManager.default.removeItem(at: envelopeFile) }
            let envelope = try RelayTransport.filePrefix(file, limit: 1024 * 1024 + 1)
            guard envelope.count <= 1024 * 1024 else { throw MoldClientError.malformedResponse }
            let object = try MoldJSON.decoder.decode(RelayTransport.Object.self, from: envelope)
            var download = URLRequest(url: try await relayObjectURL(object.url))
            download.timeoutInterval = original.timeoutInterval
            (file, answer) = try await session.download(for: download, delegate: RelayNoRedirect())
            guard let fetched = answer as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
            if !(200...299).contains(fetched.statusCode) {
                try HTTPRefusal.check(fetched, RelayTransport.filePrefix(file))
            }
            guard (200...599).contains(object.status),
                let restored = HTTPURLResponse(
                    url: original.url!, statusCode: object.status,
                    httpVersion: "HTTP/1.1", headerFields: object.headers)
            else { throw MoldClientError.malformedResponse }
            http = restored
        }
        return (file, http)
    }

    /// Preserve lazy bounded reads for retained media and SSE.
    func relayBytes(_ original: URLRequest) async throws -> (URLSession.AsyncBytes, HTTPURLResponse) {
        let (prepared, _) = try await relayPrepared(original)
        let (stream, answer) = try await session.bytes(for: prepared, delegate: relayDelegate(prepared))
        guard let http = answer as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        guard http.value(forHTTPHeaderField: "x-mold-relay-object") == "1" else { return (stream, http) }
        let envelope = try await stream.collected(upTo: 1024 * 1024)
        let object = try MoldJSON.decoder.decode(RelayTransport.Object.self, from: envelope)
        var download = URLRequest(url: try await relayObjectURL(object.url))
        download.timeoutInterval = original.timeoutInterval
        let (bytes, response) = try await session.bytes(for: download, delegate: RelayNoRedirect())
        guard let fetched = response as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        if !(200...299).contains(fetched.statusCode) { throw await streamRefusal(fetched, bytes) }
        guard (200...599).contains(object.status),
            let restored = HTTPURLResponse(
                url: original.url!, statusCode: object.status,
                httpVersion: "HTTP/1.1", headerFields: object.headers)
        else { throw MoldClientError.malformedResponse }
        return (bytes, restored)
    }
    func relayObjectURL(_ value: String) async throws -> URL {
        guard let url = URL(string: value) else { throw MoldClientError.malformedResponse }
        if HostAddress.sameOrigin(url, host.baseURL) {
            return try RelayTransport.objectURL(value, origin: host.baseURL)
        }
        guard RelayTransport.s3Identity(url) != nil else { throw MoldClientError.malformedResponse }
        let info = try await RelayTransport.cache.info(origin: host.baseURL, session: session)
        return try RelayTransport.objectURL(value, origin: host.baseURL, objectOrigin: info.objectOrigin)
    }
}
