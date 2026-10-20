import Foundation

/// In-memory admission authority; the original draft remains editable and private.
public struct PreparedVideoJob: Sendable {
    public let request: VideoJobRequest
    let release: @Sendable () async -> Void
    public init(request: VideoJobRequest, release: @escaping @Sendable () async -> Void = {}) {
        self.request = request; self.release = release
    }
    public func cancel() async { await release() }
}

public extension HTTPBackend {
    /// The ordinary reference preparation path, kept alive across read-only
    /// validation and one create. Validation must not spend its upload handles.
    func prepareVideoJob(_ original: VideoJobRequest) async throws -> PreparedVideoJob {
        let scoped = preparingLiteralVideoInputs()
        let input = original.generation
        guard input.references?.contains(where: { $0.media.authority == "upload" }) != true else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "Prepare this video again from its original references.")
        }
        let pending = input.pendingReferenceInputs != nil
        let inline = input.references?.contains(where: { $0.media.authority == "inline" }) == true
        guard pending || inline else { return .init(request: original) }
        try await requireReferenceAuthority()
        let advertised = try await capabilities().referenceUploads
        let lease: ReferenceUploadLease
        if pending {
            try VideoNormalizationPolicy.requireValidAdvertisement(advertised)
            guard let caps = advertised, VideoNormalizationPolicy.capability(caps, apiKey: host.apiKey) != nil else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_NORMALIZATION_UNAVAILABLE", "Fresh video references need a compatible keyed host.")
            }
            let instance = try await videoReferenceInstance()
            lease = try await scoped.prepareNormalizedReferences(input, capabilities: caps, expectedInstanceId: instance)
        } else if let caps = advertised, ReferenceUploadPolicy.shouldUpload(input, apiKey: host.apiKey, capabilities: caps) {
            try ReferenceUploadPolicy.validate(caps)
            let instance = try await videoReferenceInstance()
            lease = try await scoped.prepareReferenceUploads(input, capabilities: caps, expectedInstanceId: instance)
        } else {
            try ReferenceUploadPolicy.validateInlineReferences(input.references ?? [])
            let count = (input.references ?? []).filter { $0.media.authority == "inline" }.reduce(0) {
                $0 + (Data(base64Encoded: $1.media.data ?? "")?.count ?? 0)
            }
            guard count <= 32 * 1024 * 1024 else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_INLINE_TOO_LARGE", "This host accepts at most 32 MiB of inline reference media per video. Use smaller files or a host with authenticated reference uploads.")
            }
            return .init(request: original)
        }
        do {
            try await requireReferenceAuthority()
            guard lease.expiresAtMs > referenceUploadClock() else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_SESSION_INVALID", "Video preparation expired. Try again from the original references.")
            }
            var prepared = original; prepared.generation = lease.request
            let cancellation = lease.cancellation
            return .init(request: prepared, release: { await cancellation.cancel() })
        } catch { await lease.cancel(); throw error }
    }

    private func videoReferenceInstance() async throws -> String {
        guard let instance = try await status().instanceId, !instance.isEmpty,
              host.connectionInstanceID == nil || host.connectionInstanceID == instance else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_INSTANCE_MISMATCH", "The Mold instance changed before video preparation.")
        }
        return instance
    }
}
