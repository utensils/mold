import Foundation

// What a PLANNING read is allowed to carry. Split from
// `GenerateRequest+Encoding.swift` because it is a policy, not an encoding.
public extension GenerateRequest {
    /// The same request with everything a planner does not read taken out.
    ///
    /// A placement preview prices a render; it does not make one. With a 12 MB
    /// source photo attached, every keystroke in the prompt field re-uploaded
    /// ~16 MB of base64 to a machine to answer a question that never looks at
    /// it -- and the prompt, the title's tags and the collection name went
    /// with it. "A tag or a collection name ('Client X, unannounced') must not
    /// be fanned out to every candidate host just to price a render"
    /// (`studio/api/generationPlacement.ts:357-402`, ported here field for
    /// field).
    ///
    /// Media and text are BLANKED rather than dropped: whether a source image
    /// is attached is structurally relevant to placement -- it decides the
    /// conditioning path and the activation budget -- so presence is kept and
    /// only the bytes go. Filing is DELETED, because both fields are additive
    /// and absent is their normal shape; a preview files nothing anyway.
    ///
    /// The identity photographs are mold's own addition to studio's list. They
    /// are media bytes like any other, presence survives the blanking, and
    /// nothing in the planner reads a face.
    func redactedForPlacement() -> GenerateRequest {
        var redacted = self
        redacted.prompt = ""
        redacted.negativePrompt = Self.blanked(negativePrompt)
        redacted.originalPrompt = Self.blanked(originalPrompt)
        redacted.sourceImage = Self.blanked(sourceImage)
        redacted.sourceImageName = Self.blanked(sourceImageName)
        redacted.maskImage = Self.blanked(maskImage)
        redacted.controlImage = Self.blanked(controlImage)
        redacted.audioFile = Self.blanked(audioFile)
        redacted.sourceVideo = Self.blanked(sourceVideo)
        redacted.extendVideo = Self.blanked(extendVideo)
        redacted.idImage = Self.blanked(idImage)
        redacted.idImageName = Self.blanked(idImageName)
        redacted.idImages = idImages.map { $0.map { _ in "" } }
        redacted.idImageNames = idImageNames.map { $0.map { _ in "" } }
        redacted.editImages = editImages.map { $0.map { _ in "" } }
        redacted.keyframes = keyframes?.map {
            KeyframeCondition(frame: $0.frame, image: "", name: $0.name)
        }
        redacted.tags = nil
        redacted.collection = nil
        return redacted
    }

    private static func blanked(_ value: String?) -> String? {
        value == nil ? nil : ""
    }
}
