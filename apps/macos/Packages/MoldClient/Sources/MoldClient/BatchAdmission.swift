import Foundation

/// A batch is one atomic admission of up to 64 ordered children. There is no
/// separate "single render" path on the server -- a one-off is a batch of one.
public struct BatchAdmission: Codable, Sendable {
    /// Minted on the device and PERSISTED BEFORE SENDING. This is the
    /// idempotency fence: if the response is lost, the work is recovered by
    /// asking the host about this id, never by submitting again.
    public let clientBatchId: String
    public let requests: [GenerateRequest]
    /// A one-use handle authorising the HOST to hydrate this admission from
    /// the print's own retained source media.
    ///
    /// It rides the `x-mold-retained-media-session` HEADER, never the body,
    /// which is what `CodingKeys` below is for: it is a credential, so it
    /// must not reach a log, a persisted draft, or a recovery record, and
    /// omitting it from the coding keys makes that structural rather than
    /// something each writer has to remember.
    public var retainedMediaSession: String?

    private enum CodingKeys: String, CodingKey {
        case clientBatchId, requests
    }

    public init(clientBatchId: String = UUID().uuidString, requests: [GenerateRequest],
                retainedMediaSession: String? = nil) {
        self.clientBatchId = clientBatchId
        self.requests = requests
        self.retainedMediaSession = retainedMediaSession
    }

    /// The host binds a session to exactly ONE child, because it cannot know
    /// which of several the media belongs to
    /// (`validate_reuse_batch_cardinality`, `gallery_source_media.rs:28-42`).
    /// Its own sentence and its own code, so a client and the host refuse in
    /// the same words.
    public var retainedMediaBatchRefusal: MoldClientError? {
        guard retainedMediaSession != nil, requests.count != 1 else { return nil }
        return .http(status: 422, code: "RETAINED_MEDIA_REUSE_BATCH_AMBIGUOUS",
                     message: "a retained-media reuse session binds exactly one batch child")
    }
}
