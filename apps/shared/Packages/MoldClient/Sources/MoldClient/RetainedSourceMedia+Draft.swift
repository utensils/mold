import Foundation

public extension RetainedSourceMedia {
    /// Typed references already have visible descriptors. Legacy inputs need
    /// ordinary authoring bytes so the wells and outgoing request agree.
    static let materializableFields: Set<Field> = [
        .sourceImage, .maskImage, .editImages, .identityImage, .controlImage,
        .keyframes, .audioFile, .sourceVideo, .extendVideo
    ]

    static func draftField(for member: Member) -> Field? {
        let field = fieldForRole[member.role]
        return field == .identityImages ? .identityImage : field
    }

    static func vacantDraftFields(in media: DraftMedia) -> Set<Field> {
        var vacant = materializableFields
        if media.sourceImage != nil || media.sourceImageOriginal != nil || media.parked.sourceImage != nil { vacant.remove(.sourceImage) }
        if media.maskImage != nil || media.parked.maskImage != nil { vacant.remove(.maskImage) }
        if !media.editImages.isEmpty || !media.parked.editImages.isEmpty { vacant.remove(.editImages) }
        if !(media.identity?.photos.isEmpty ?? true) || !(media.parked.identity?.photos.isEmpty ?? true) { vacant.remove(.identityImage) }
        if media.control?.image != nil || media.parked.control?.image != nil { vacant.remove(.controlImage) }
        if !media.keyframes.isEmpty || !media.parked.keyframes.isEmpty || media.boundaryKeyframes.values.contains(where: { !$0.isEmpty }) { vacant.remove(.keyframes) }
        if media.audioFile != nil || media.parked.audioFile != nil { vacant.remove(.audioFile) }
        if media.sourceVideo != nil || media.parked.sourceVideo != nil { vacant.remove(.sourceVideo) }
        if media.extendVideo != nil || media.parked.extendVideo != nil { vacant.remove(.extendVideo) }
        return vacant
    }

    /// Tracks attachment mutations independently of scalar edits. A caller
    /// increments these roles' revisions, including attach-then-remove edits.
    static func changedDraftFields(from old: DraftMedia, to new: DraftMedia) -> Set<Field> {
        var fields: Set<Field> = []
        if old.sourceImage != new.sourceImage || old.sourceImageOriginal != new.sourceImageOriginal || old.parked.sourceImage != new.parked.sourceImage { fields.insert(.sourceImage) }
        if old.maskImage != new.maskImage || old.parked.maskImage != new.parked.maskImage { fields.insert(.maskImage) }
        if old.editImages != new.editImages || old.parked.editImages != new.parked.editImages { fields.insert(.editImages) }
        if old.identity?.photos != new.identity?.photos || old.parked.identity?.photos != new.parked.identity?.photos { fields.insert(.identityImage) }
        if old.control?.image != new.control?.image || old.parked.control?.image != new.parked.control?.image { fields.insert(.controlImage) }
        if old.keyframes != new.keyframes || old.parked.keyframes != new.parked.keyframes || old.boundaryKeyframes != new.boundaryKeyframes { fields.insert(.keyframes) }
        if old.audioFile != new.audioFile || old.parked.audioFile != new.parked.audioFile { fields.insert(.audioFile) }
        if old.sourceVideo != new.sourceVideo || old.parked.sourceVideo != new.parked.sourceVideo { fields.insert(.sourceVideo) }
        if old.extendVideo != new.extendVideo || old.parked.extendVideo != new.parked.extendVideo { fields.insert(.extendVideo) }
        return fields
    }

    /// Decode before publishing any attachment. Keyframes are JSON documents,
    /// not pictures; their exact order and pixel-frame indices must survive.
    /// Source and its mask land together, never over a replacement picture.
    static func materializedDraft(
        _ downloaded: [(member: Member, bytes: Data)], into draft: RenderDraft
    ) throws -> RenderDraft {
        let request = GenerateRequest(prompt: "", model: "", width: draft.width,
                                      height: draft.height, steps: draft.steps, guidance: draft.guidance)
        if downloaded.contains(where: { $0.member.role == "identity_image" }) && downloaded.contains(where: { $0.member.role == "identity_images" }) {
            throw RelayFailure.ambiguous(.identityImage)
        }
        let decoded = try relayed(downloaded, into: request)
        let vacant = vacantDraftFields(in: draft.media)
        let members = Dictionary(grouping: downloaded.map(\.member), by: { draftField(for: $0) })
        func name(_ field: Field) -> String? { members[field]?.first?.displayName }
        var restored = draft
        if vacant.contains(.sourceImage), let source = decoded.sourceImage {
            restored.media.sourceImage = source
            restored.media.sourceImageOriginal = source
            restored.media.sourceImageName = name(.sourceImage)
            restored.media.sourceImageOriginalName = name(.sourceImage)
            restored.media.sourceImagePixels = ReferenceCanvas.uprightPixels(ofBase64: source)
            if vacant.contains(.maskImage) { restored.media.maskImage = decoded.maskImage }
        }
        if vacant.contains(.editImages), let images = decoded.editImages { restored.media.editImages = images }
        if vacant.contains(.identityImage) {
            let photos = decoded.idImages ?? decoded.idImage.map { [$0] }
            if let photos {
                var identity = restored.media.identity ?? restored.media.parked.identity ?? .init(photos: [])
                identity.photos = zip(photos, members[.identityImage] ?? []).map { .init(encoded: $0, name: $1.displayName) }
                restored.media.identity = identity
                restored.media.parked.identity = nil
            }
        }
        if vacant.contains(.controlImage), let image = decoded.controlImage {
            var control = restored.media.control ?? restored.media.parked.control ?? .init()
            control.image = image
            restored.media.control = control
            restored.media.parked.control = nil
        }
        if vacant.contains(.keyframes), let keyframes = decoded.keyframes { restored.media.keyframes = keyframes }
        if vacant.contains(.audioFile), let audio = decoded.audioFile {
            restored.media.audioFile = audio
            restored.media.audioFileName = name(.audioFile)
        }
        if vacant.contains(.sourceVideo), let video = decoded.sourceVideo {
            restored.media.sourceVideo = video
            restored.media.sourceVideoName = name(.sourceVideo)
        }
        if vacant.contains(.extendVideo), let video = decoded.extendVideo {
            restored.media.extendVideo = video
            restored.media.extendVideoName = name(.extendVideo)
        }
        return restored
    }
}
