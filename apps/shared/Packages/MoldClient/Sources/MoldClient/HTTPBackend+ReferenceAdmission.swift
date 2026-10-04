import Foundation

extension HTTPBackend {
    func submitWithReferenceUploads(_ admission: BatchAdmission) async throws -> BatchStatus {
        // Drafts and persisted recovery records must carry original authority, never spent handles.
        guard !admission.requests.contains(where: { $0.references?.contains(where: { $0.media.authority == "upload" }) == true }) else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "Retry reference admission from the original media snapshot.")
        }
        guard admission.requests.contains(where: { $0.references?.contains(where: {
            $0.media.authority == "inline" && ["image", "audio", "video"].contains($0.kind)
        }) == true }) else { return try await postAdmission(admission) }
        let caps = try await capabilities().referenceUploads
        guard let caps, admission.requests.contains(where: {
            ReferenceUploadPolicy.shouldUpload($0, apiKey: host.apiKey, capabilities: caps)
        }) else {
            for request in admission.requests {
                try ReferenceUploadPolicy.validateInlineReferences(request.references ?? [])
                let bytes = (request.references ?? []).filter { $0.media.authority == "inline" }.reduce(0) {
                    $0 + (Data(base64Encoded: $1.media.data ?? "")?.count ?? 0)
                }
                guard bytes <= 32 * 1024 * 1024 else {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_INLINE_TOO_LARGE", "This host accepts at most 32 MiB of inline reference media per render. Choose smaller files or connect to a host with authenticated reference uploads.")
                }
            }
            return try await postAdmission(admission)
        }
        try ReferenceUploadPolicy.validate(caps)
        let uploadCount = admission.requests.filter {
            ReferenceUploadPolicy.shouldUpload($0, apiKey: host.apiKey, capabilities: caps)
        }.count
        guard uploadCount <= caps.maxActiveSessions! else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_SESSION_LIMIT", "Split this batch into chunks within the host's active reference-session limit.")
        }
        guard let instance = try await status().instanceId, !instance.isEmpty else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_INSTANCE_REQUIRED", "The host did not provide an exact instance identity.")
        }
        if let expected = host.connectionInstanceID, instance != expected {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_INSTANCE_MISMATCH", "The Mold instance changed before reference admission.")
        }
        var leases: [ReferenceUploadLease] = []
        do {
            var requests: [GenerateRequest] = []
            for request in admission.requests {
                try Task.checkCancellation()
                if ReferenceUploadPolicy.shouldUpload(request, apiKey: host.apiKey, capabilities: caps) {
                    let lease = try await prepareReferenceUploads(request, capabilities: caps, expectedInstanceId: instance)
                    leases.append(lease); requests.append(lease.request)
                } else { requests.append(request) }
            }
            try Task.checkCancellation()
            guard leases.allSatisfy({ $0.expiresAtMs > Self.referenceUploadNow }) else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_SESSION_INVALID", "A reference session expired before batch admission.")
            }
            let staged = BatchAdmission(clientBatchId: admission.clientBatchId, requests: requests,
                                        retainedMediaSession: admission.retainedMediaSession)
            let result = try await postAdmission(staged)
            for lease in leases { await lease.cancel() }
            return result
        } catch {
            for lease in leases { await lease.cancel() }
            throw error
        }
    }
}
