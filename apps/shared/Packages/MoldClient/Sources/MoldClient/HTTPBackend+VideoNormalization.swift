import Foundation

public extension HTTPBackend {
    /// A single attempt checks its captured full authority after every asynchronous boundary.
    /// This closure never enters the request, candidate fingerprint or recovery snapshot.
    func checkingReferenceAuthority(_ isCurrent: @escaping @Sendable () async -> Bool) -> Self {
        var copy = self; copy.referenceAuthority = isCurrent; return copy
    }

    func prepareNormalizedReferences(_ original: GenerateRequest, capabilities uploads: ReferenceUploadCapabilities,
                                     expectedInstanceId: String) async throws -> ReferenceUploadLease {
        try await requireReferenceAuthority()
        let captured = try VideoNormalizationCapture.capture(original, capabilities: uploads, apiKey: host.apiKey)
        let caps = uploads.videoNormalization!
        let mismatch = ReferenceUploadPolicy.refusal("REFERENCE_NORMALIZATION_RESPONSE_MISMATCH", "The host returned mismatched normalization authority.")
        guard !expectedInstanceId.isEmpty else { throw mismatch }
        var opening = request(caps.candidateSessionPath!)
        opening.httpMethod = "POST"; opening.httpBody = captured.body
        opening.setValue("application/json", forHTTPHeaderField: "Content-Type")
        opening.setValue(captured.fingerprint, forHTTPHeaderField: caps.candidateDigestHeader!)
        let bytes = try await referenceUploadBytes(opening)
        var cancellation: ReferenceUploadCancellation?
        // Salvage only a syntactically safe bearer from a malformed reply, exclusively for release.
        if let row = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
           let handle = row["session_handle"] as? String, ReferenceUploadPolicy.secret(handle) {
            cancellation = .init(backend: self, path: uploads.sessionPath!, header: uploads.sessionHandleHeader!, handle: handle)
        }
        do {
            try await requireReferenceAuthority()
            let session = try decoded(VideoNormalizationSession.self, from: bytes, route: caps.candidateSessionPath!)
            let indices = captured.entries.filter { $0.bytes != nil }.map(\.index)
            guard let cancellation, session.protocolVersion == 1, session.instanceId == expectedInstanceId,
                  ReferenceUploadPolicy.secret(session.sessionId), session.candidateScopeSha256 == captured.fingerprint,
                  !session.sessionComplete, session.expiresAtMs > referenceUploadClock(),
                  Set(session.uploads.map(\.reference)) == Set(indices), session.uploads.count == indices.count,
                  Set(session.uploads.map(\.handle)).count == indices.count,
                  session.uploads.allSatisfy({ ReferenceUploadPolicy.secret($0.handle) }) else { throw mismatch }
            var canonical: [GenerationReference] = []
            var receipts: [VideoNormalizationFinalizeReceipt] = []
            var total = 0
            for entry in captured.entries {
                try await requireReferenceAuthority()
                guard session.expiresAtMs > referenceUploadClock() else { throw mismatch }
                guard let data = entry.bytes else { canonical.append(entry.original.canonicalReference!); continue }
                guard let slot = session.uploads.first(where: { $0.reference == entry.index }) else { throw mismatch }
                var upload = request(uploads.uploadPath!)
                upload.httpMethod = "PUT"; upload.httpBody = data; upload.timeoutInterval = 300
                upload.setValue(slot.handle, forHTTPHeaderField: uploads.uploadHandleHeader!)
                upload.setValue(entry.mimeType, forHTTPHeaderField: "Content-Type")
                let body = try await referenceUploadBytes(upload)
                try await requireReferenceAuthority()
                let result = try decoded(VideoNormalizationUpload.self, from: body, route: uploads.uploadPath!)
                guard result.protocolVersion == 1, result.instanceId == expectedInstanceId,
                      result.sessionId == session.sessionId, result.candidateScopeSha256 == captured.fingerprint,
                      result.reference == entry.index, !result.sessionComplete else { throw mismatch }
                var reference = try result.receipt.canonical(metadata: result.metadata, entry: entry, capabilities: caps)
                guard result.receipt.canonicalByteLength <= caps.maxSessionBytes! - total else { throw mismatch }
                total += result.receipt.canonicalByteLength
                reference.media = .init(authority: "upload", handle: slot.handle)
                canonical.append(reference)
                receipts.append(.init(reference: entry.index, kind: reference.kind,
                    originalSha256: entry.digest, canonicalSha256: result.receipt.canonicalSha256))
            }
            struct Finalize: Encodable {
                let protocolVersion = 1
                let instanceId: String; let sessionId: String; let candidateScopeSha256: String
                let receipts: [VideoNormalizationFinalizeReceipt]
            }
            try await requireReferenceAuthority()
            guard session.expiresAtMs > referenceUploadClock() else { throw mismatch }
            var finalize = request(caps.finalizePath!)
            finalize.httpMethod = "POST"; finalize.setValue("application/json", forHTTPHeaderField: "Content-Type")
            finalize.setValue(session.sessionHandle, forHTTPHeaderField: uploads.sessionHandleHeader!)
            finalize.httpBody = try MoldJSON.encoder.encode(Finalize(instanceId: expectedInstanceId,
                sessionId: session.sessionId, candidateScopeSha256: captured.fingerprint, receipts: receipts))
            let finalizedBytes = try await referenceUploadBytes(finalize)
            try await requireReferenceAuthority()
            let finalized = try decoded(VideoNormalizationFinal.self, from: finalizedBytes, route: caps.finalizePath!)
            guard finalized.protocolVersion == 1, finalized.instanceId == expectedInstanceId,
                  finalized.sessionId == session.sessionId, finalized.candidateScopeSha256 == captured.fingerprint,
                  finalized.sessionComplete, ReferenceUploadPolicy.digest(finalized.requestScopeSha256),
                  finalized.receipts == receipts, finalized.references.count == canonical.count,
                  session.expiresAtMs > referenceUploadClock() else { throw mismatch }
            for (returned, expected) in zip(finalized.references, canonical) {
                guard returned.media == .init(authority: "descriptor"),
                      normalizedDescriptor(returned) == normalizedDescriptor(expected) else { throw mismatch }
            }
            var final = original; final.pendingReferenceInputs = nil; final.references = canonical
            return ReferenceUploadLease(request: final, expiresAtMs: session.expiresAtMs,
                requestScopeSha256: finalized.requestScopeSha256, cancellation: cancellation)
        } catch { await cancellation?.cancel(); throw error }
    }
}

extension HTTPBackend {
    func requireReferenceAuthority() async throws {
        try Task.checkCancellation()
        if let referenceAuthority, !(await referenceAuthority()) { throw CancellationError() }
        try Task.checkCancellation()
    }
    private func normalizedDescriptor(_ value: GenerationReference) -> GenerationReference {
        var copy = value; copy.media = .init(authority: "descriptor")
        if copy.kind == "video", copy.hasAudio == nil { copy.hasAudio = false }
        return copy
    }
}
