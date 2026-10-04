import Foundation

public extension HTTPBackend {
    /// Upload a request snapshot into a fresh session. Never reuse this lease for a sibling or retry.
    func prepareReferenceUploads(_ original: GenerateRequest, capabilities caps: ReferenceUploadCapabilities,
                                 expectedInstanceId: String) async throws -> ReferenceUploadLease {
        try Task.checkCancellation()
        try ReferenceUploadPolicy.validate(caps)
        guard !(host.apiKey?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true),
              !expectedInstanceId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_AUTH_REQUIRED", "Reference uploads require a keyed host and exact instance identity.")
        }
        let captured = try ReferenceUploadCapture.capture(original, capabilities: caps)
        var scoped = original
        scoped.references = captured.map(\.descriptor)
        struct SessionBody: Encodable { let request: GenerateRequest; let uploadReferences: [Int] }
        var opening = request(caps.sessionPath!)
        opening.httpMethod = "POST"
        opening.setValue("application/json", forHTTPHeaderField: "Content-Type")
        opening.httpBody = try MoldJSON.encoder.encode(SessionBody(request: scoped, uploadReferences: captured.filter { $0.bytes != nil }.map(\.index)))
        let sessionData = try await referenceUploadBytes(opening)
        var cancellation: ReferenceUploadCancellation?
        // Salvage only a syntactically valid credential for best-effort cleanup of malformed replies.
        if let row = try? JSONSerialization.jsonObject(with: sessionData) as? [String: Any],
           let handle = row["session_handle"] as? String, ReferenceUploadPolicy.secret(handle) {
            cancellation = .init(backend: self, path: caps.sessionPath!, header: caps.sessionHandleHeader!, handle: handle)
        }
        do {
            let session = try decoded(ReferenceUploadSessionResponse.self, from: sessionData, route: caps.sessionPath!)
            try session.validate(instance: expectedInstanceId, indices: captured.filter { $0.bytes != nil }.map(\.index), now: Self.referenceUploadNow)
            guard let cancellation else { throw MoldClientError.malformedResponse }
            var final = original
            var scope = session.requestScopeSha256.lowercased()
            var completed = 0
            var canonical: [GenerationReference] = []
            for entry in captured {
                try Task.checkCancellation()
                guard let bytes = entry.bytes else { canonical.append(entry.original); continue }
                guard let slot = session.uploads.first(where: { $0.reference == entry.index }) else { throw MoldClientError.malformedResponse }
                var upload = request(caps.uploadPath!)
                upload.httpMethod = "PUT"
                upload.timeoutInterval = 300
                upload.setValue(slot.handle, forHTTPHeaderField: caps.uploadHandleHeader!)
                upload.setValue(entry.original.mimeType, forHTTPHeaderField: "Content-Type")
                upload.httpBody = bytes
                let response = try decoded(ReferenceUploadCompletion.self, from: await referenceUploadBytes(upload), route: caps.uploadPath!)
                completed += 1
                guard response.instanceId == expectedInstanceId, response.reference == entry.index,
                      ReferenceUploadPolicy.digest(response.requestScopeSha256),
                      response.sessionComplete == (completed == session.uploads.count) else {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_RESPONSE_MISMATCH", "The host returned mismatched upload completion.")
                }
                var ref = try response.metadata.canonical(original: entry.original, index: entry.index)
                ref.media = .init(authority: "upload", handle: slot.handle)
                canonical.append(ref)
                scope = response.requestScopeSha256.lowercased()
            }
            try Task.checkCancellation()
            guard session.expiresAtMs > Self.referenceUploadNow else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_SESSION_INVALID", "The reference-upload session expired.")
            }
            final.references = canonical
            return ReferenceUploadLease(request: final, expiresAtMs: session.expiresAtMs, requestScopeSha256: scope, cancellation: cancellation)
        } catch {
            await cancellation?.cancel()
            throw error
        }
    }
}

extension HTTPBackend {
    static var referenceUploadNow: Int64 { Int64(Date().timeIntervalSince1970 * 1000) }
    /// Credential-bearing upload routes never follow redirects, even for plain HTTP hosts.
    func referenceUploadBytes(_ original: URLRequest) async throws -> Data {
        let (prepared, _) = try await relayPrepared(original)
        let (data, response) = try await session.data(for: prepared, delegate: RelayNoRedirect())
        guard let http = response as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        let (resolved, answer) = try await relayObject(data, response: http, original: original)
        try HTTPRefusal.check(answer, resolved)
        return resolved
    }
}
