import Foundation

extension HTTPBackend {
    /// Finalized requests stay byte-for-byte unchanged between preparation and admission.
    /// A refusal releases every unused lease; callers retain only the original draft bytes.
    func submitNormalizedAdmission(_ admission: BatchAdmission) async throws -> BatchStatus {
        try await requireReferenceAuthority()
        let advertised = try await capabilities().referenceUploads
        try await requireReferenceAuthority()
        try VideoNormalizationPolicy.requireValidAdvertisement(advertised)
        guard let caps = advertised, VideoNormalizationPolicy.capability(caps, apiKey: host.apiKey) != nil,
              admission.requests.count <= (caps.maxActiveSessions ?? 0) else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_NORMALIZATION_UNAVAILABLE", "Fresh video references need a compatible keyed host and a batch within its reference-session limit.")
        }
        let instance = try await status().instanceId
        try await requireReferenceAuthority()
        guard let instance, !instance.isEmpty,
              host.connectionInstanceID == nil || host.connectionInstanceID == instance else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_INSTANCE_MISMATCH", "The Mold instance changed before video preparation.")
        }
        var leases: [ReferenceUploadLease] = []
        do {
            var requests: [GenerateRequest] = []
            for request in admission.requests {
                let lease: ReferenceUploadLease
                if request.pendingReferenceInputs != nil {
                    lease = try await prepareNormalizedReferences(request, capabilities: caps, expectedInstanceId: instance)
                } else {
                    lease = try await prepareReferenceUploads(request, capabilities: caps, expectedInstanceId: instance)
                }
                leases.append(lease); requests.append(lease.request)
            }
            try await requireReferenceAuthority()
            // No candidate is priced until all exact canonical facts exist. This preview is on
            // the captured normalizer; it never fans a one-use handle out to other machines.
            if let first = requests.first {
                let placement = try await placementPreview(first.redactedForPlacement(), copies: requests.count)
                if let licence = placement.outstandingLicenses.first {
                    throw MoldClientError.licenseRequired(licence, mismatch: false)
                }
            }
            try await requireReferenceAuthority()
            guard leases.allSatisfy({ $0.expiresAtMs > referenceUploadClock() }) else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_SESSION_INVALID", "Video preparation expired before admission. Try again from the original references.")
            }
            let final = BatchAdmission(clientBatchId: admission.clientBatchId, requests: requests,
                retainedMediaSession: admission.retainedMediaSession)
            let accepted = try await postAdmission(final)
            for lease in leases { await lease.cancel() }
            return accepted
        } catch {
            for lease in leases { await lease.cancel() }
            throw error
        }
    }
}
