import Foundation

/// What the Generate pane was holding when the app last quit.
///
/// A DESCRIPTOR, not a draft: every byte-bearing root is excluded exactly as
/// `sanitizePersistedForm` excludes them (`useGenerateForm.ts:221-268`) -- the
/// source picture, the references, the mask, the identity photographs, the
/// control picture, the conditioning audio, the source and extend video, and
/// every keyframe image. Studio keeps their descriptors because its bytes
/// live in IndexedDB under the draft id; this app has no such store, so it
/// keeps NOTHING about them. A restored draft that named a picture it could
/// not produce would be a draft claiming media it no longer has.
///
/// `CodingKeys` is spelled out and this is read and written with
/// `MoldJSON.local*`, which has NO key strategy: the wire pair are not
/// inverses (`baseURL` -> `base_url` -> `baseUrl`) and this is exactly the
/// kind of file that comes back empty because of it -- the `StoredHost` trap.
public struct DraftDescriptor: Codable, Hashable, Sendable {
    /// Bumped whenever a field's MEANING changes. A mismatch discards the
    /// whole descriptor rather than merging: a half-restored draft is worse
    /// than an empty one, because nothing on screen says which half is real.
    public static let currentVersion = 1

    public var version: Int = DraftDescriptor.currentVersion
    /// What was chosen, so the pane can put the model back. A model that is
    /// gone simply does not match anything, and the rest of the draft still
    /// restores. The MACHINE is deliberately not here: `MachineChoiceStore`
    /// already persists it, and a second copy would be a second authority.
    public var model: String?
    public var family: String?
    public var recipeID: String?

    public var prompt: String
    public var negativePrompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Double
    public var batchSize: Int
    public var seed: UInt64?
    public var locksSeed: Bool
    public var frames: Int?
    public var fps: Int?
    public var pipeline: String?
    public var enableAudio: Bool
    public var videoOnly: Bool
    public var strength: Double
    public var title: String
    public var tags: [String]
    public var collectionName: String?
    public var autoTagTitle: Bool
    public var outputFormat: String?
    public var upscaleModel: String?
    public var savesToGallery: Bool
    public var canvasIntent: CanvasIntent
    /// A policy, not pixels -- it describes what WOULD be done to a picture
    /// somebody attaches next.
    public var sourceFit: SourceFit

    public var scheduler: String?
    public var cfgPlus: Bool
    public var sampleShift: Double?
    public var distillStrengthHigh: Double?
    public var distillStrengthLow: Double?
    public var stgScale: Double?
    public var stgBlocks: String
    public var rescaleScale: Double?
    public var modalityScale: Double?
    public var skipStep: Int?

    enum CodingKeys: String, CodingKey {
        case version, model, family, recipeID, prompt, negativePrompt, width, height
        case steps, guidance, batchSize, seed, locksSeed, frames, fps, pipeline
        case enableAudio, videoOnly, strength, title, tags, collectionName, autoTagTitle
        case outputFormat, upscaleModel, savesToGallery, canvasIntent, sourceFit
        case scheduler, cfgPlus, sampleShift, distillStrengthHigh, distillStrengthLow
        case stgScale, stgBlocks, rescaleScale, modalityScale, skipStep
    }
}
