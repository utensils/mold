import Foundation

/// Explicit destinations in the current recipe. Each choice stages a new authored input.
public enum SourceMediaAttachment: String, Identifiable, CaseIterable, Sendable {
    case source, reference, firstFrame, lastFrame, audio, video, continuation, front, left, back, right
    public var id: Self { self }
    public var title: String {
        switch self {
        case .front: "Use as front view"
        case .left: "Use as left view"
        case .back: "Use as back view"
        case .right: "Use as right view"
        case .source: "Use as source image"
        case .reference: "Add as reference"
        case .firstFrame: "Use as first frame"
        case .lastFrame: "Use as last frame"
        case .audio: "Use as audio"
        case .video: "Use as source video"
        case .continuation: "Continue this video"
        }
    }
    public static func choices(member: RetainedSourceMedia.Member, recipe: GenerationRecipe?, media: DraftMedia) -> [Self] {
        guard let recipe else { return [] }
        let caps = recipe.capabilities
        if ["audio", "video"].contains(member.contentKind), let typed = caps.generationReferences, typed.mode.isVisible {
            guard typed.kinds.contains(member.contentKind), media.generationReferences.count < typed.maxCount,
                  media.generationReferences.filter({ $0.kind == member.contentKind }).count < (member.contentKind == "audio" ? typed.maxAudios : typed.maxVideos) else { return [] }
            return [.reference]
        }
        switch member.contentKind {
        case "image":
            if acceptsEndpoints(capabilities: caps) { return [.firstFrame, .lastFrame] }
            if let named = caps.mesh?.namedViews, named.mode.isVisible {
                return named.roles.compactMap { Self(rawValue: $0.rawValue) }
            }
            if let typed = caps.generationReferences, typed.mode.isVisible {
                return typed.kinds.contains("image") && media.generationReferences.count < typed.maxCount && media.generationReferences.filter({ $0.kind == "image" }).count < typed.maxImages ? [.reference] : []
            }
            var result: [Self] = []
            if caps.readsSourceImage { result.append(.source) }
            if let references = caps.referenceImages(family: nil, model: nil), references.hasRoom(for: media.editImages.count) { result.append(.reference) }
            return result
        case "audio": return caps.acceptsSourceAudio ? [.audio] : []
        case "video": return (caps.acceptsSourceVideo ? [.video] : []) + (caps.supportsExtend == true ? [.continuation] : [])
        default: return []
        }
    }
    public func applying(data: Data, member: RetainedSourceMedia.Member, to draft: RenderDraft, recipe: GenerationRecipe) async throws -> RenderDraft {
        guard Self.choices(member: member, recipe: recipe, media: draft.media).contains(self) else {
            throw MoldClientError.unreachable("This model cannot use that source in the selected role.")
        }
        var result = draft
        if self == .reference && member.contentKind != "image" {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(member.contentKind == "audio" ? "wav" : "mp4")
            defer { try? FileManager.default.removeItem(at: url) }
            try data.write(to: url, options: .atomic)
            var reference = try await GenerationReferenceImporter.load(url: url)
            reference.provenance?.name = member.displayName
            result.media.appendGenerationReference(reference)
            if let error = result.media.generationReferenceError(capabilities: recipe.capabilities, allowIncomplete: true) { throw MoldClientError.unreachable(error) }
            return result
        }
        switch self {
        case .audio:
            result.media.audioFile = data.base64EncodedString(); result.media.audioFileName = member.displayName
        case .video:
            result.media.sourceVideo = data.base64EncodedString(); result.media.sourceVideoName = member.displayName
        case .continuation:
            result.media.settingExtend(video: data.base64EncodedString(), name: member.displayName)
        default:
            let picture = try await PictureImport.conforming(data, name: member.displayName, accepting: PictureImport.engineReadable)
            switch self {
            case .source: DraftPictureAttachment.useAsSource(picture, in: &result, recipe: recipe)
            case .firstFrame, .lastFrame:
                Self.setEndpoint(first: self == .firstFrame, picture: picture, draft: &result, recipe: recipe)
            case .front, .left, .back, .right:
                let role = GenerationImageReferenceRole(rawValue: rawValue)!
                result.media.generationReferences.removeAll { $0.role == role }
                result.media.appendGenerationReference(try GenerationReferenceImporter.image(picture, role: role))
                if let error = result.media.generationReferenceError(capabilities: recipe.capabilities, allowIncomplete: true) { throw MoldClientError.unreachable(error) }
            case .reference:
                if recipe.capabilities.generationReferences?.mode.isVisible == true || recipe.capabilities.mesh?.namedViews?.mode.isVisible == true {
                    result.media.appendGenerationReference(try GenerationReferenceImporter.image(picture))
                    if let error = result.media.generationReferenceError(capabilities: recipe.capabilities, allowIncomplete: true) { throw MoldClientError.unreachable(error) }
                } else if let cap = recipe.capabilities.referenceImages(family: nil, model: nil) {
                    DraftPictureAttachment.addReference(picture, to: &result, capability: cap, recipe: recipe)
                }
            default: break
            }
        }
        return result
    }
    public static func acceptsEndpoints(capabilities: RecipeCapabilities) -> Bool {
        BoundaryFramePolicy.resolve(capabilities: capabilities) != nil || capabilities.acceptsKeyframes
    }

    /// LTX interpolation replaces only the selected endpoint. Unlike a paired
    /// boundary recipe, its interior authored anchors remain meaningful.
    public static func setEndpoint(first: Bool, picture: ImportedPicture?, draft: inout RenderDraft, recipe: GenerationRecipe) {
        let caps = recipe.capabilities
        if BoundaryFramePolicy.resolve(capabilities: caps) != nil {
            BoundaryFramePolicy.set(first: first, picture: picture, draft: &draft, capabilities: caps, recipe: recipe)
            return
        }
        guard caps.acceptsKeyframes else { return }
        let frames = max(draft.frames ?? recipe.defaults.frames ?? 2, 2)
        let index = first ? 0 : frames - 1
        draft.media.keyframes.removeAll { $0.frame == index }
        draft.media.parked.keyframes.removeAll { $0.frame == index }
        if let picture {
            draft.media.addingKeyframe(.init(frame: index, image: picture.encoded, name: picture.name))
            draft.frames = frames
            draft.media.reconcileSourceImage(supported: false)
            draft.media.reconcileMask(supported: false)
            if let pixels = PictureImport.pixelSize(of: picture.data) {
                draft.attachSourceShape(pixels, recipe: recipe, replaced: false)
            }
        }
    }

}
