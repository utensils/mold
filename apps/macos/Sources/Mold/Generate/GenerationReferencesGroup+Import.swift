import AppKit
import MoldClient
import UniformTypeIdentifiers

extension GenerationReferencesGroup {
    func attachPicture(_ picture: ImportedPicture, replacing: Int? = nil,
                               role: GenerationImageReferenceRole? = nil, expected: GenerationReference? = nil, session: ReferenceImportSession) {
        guard session.isCurrent(controller: controller, media: draft.media) else { return }
        if let replacing, let expected {
            guard draft.media.generationReferences.indices.contains(replacing),
                  draft.media.generationReferences[replacing] == expected else { return }
        }
        do {
            let reference = try GenerationReferenceImporter.image(picture, role: role)
            var candidate = draft.media
            if let replacing { candidate.replaceGenerationReference(at: replacing, with: reference) }
            else { candidate.appendGenerationReference(reference) }
            if let reason = candidate.generationReferenceError(capabilities: recipe.capabilities, allowIncomplete: true) {
                failure = reason; return
            }
            draft.media = candidate
            session.advance(controller: controller, media: draft.media)
            failure = nil
        } catch { failure = error.failureSentence }
    }

    func choose(kind: String, replacing: Int? = nil, role: GenerationImageReferenceRole? = nil) {
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        let panel = NSOpenPanel()
        panel.allowedContentTypes = kind == "video" ? [.movie] : kind == "audio" ? [.audio] : [.image]
        panel.allowsMultipleSelection = replacing == nil && role == nil
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        let expected = replacing.flatMap { draft.media.generationReferences.indices.contains($0)
            ? draft.media.generationReferences[$0] : nil }
        importing = true
        importTask?.cancel()
        importTask = Task {
            defer { importing = false }
            do {
                for url in urls {
                    let reference = try await GenerationReferenceImporter.load(url: url, role: role)
                    guard !Task.isCancelled, session.isCurrent(controller: controller, media: draft.media) else { return }
                    if let replacing, let expected {
                        guard draft.media.generationReferences.indices.contains(replacing),
                              draft.media.generationReferences[replacing] == expected else { return }
                    }
                    guard role != nil || reference.kind == kind else {
                        failure = "Choose a \(kind) file."; return
                    }
                    var candidate = draft.media
                    if let replacing { candidate.replaceGenerationReference(at: replacing, with: reference) }
                    else { candidate.appendGenerationReference(reference) }
                    if let reason = candidate.generationReferenceError(capabilities: recipe.capabilities, allowIncomplete: true) {
                        failure = reason; return
                    }
                    draft.media = candidate
                    session.advance(controller: controller, media: draft.media)
                }
                failure = nil
            } catch { failure = error.failureSentence }
        }
    }
}
