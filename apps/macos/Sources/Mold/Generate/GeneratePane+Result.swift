import AppKit
import MoldClient
import SwiftUI

// Building the finished result's verbs once, where the draft and the chosen
// recipe are both in hand. `RunCanvas` is a view over a `RunState` and holds
// neither.
extension GeneratePane {
    var resultActions: ResultActions {
        ResultActions(
            save: { result in Task { await saveResult(result) } },
            copy: { result in Task { await copyResult(result) } },
            showInLibrary: { destination = .library },
            useAsSource: canReuseAsSource ? { result in
                Task { await attachResult(result) { picked, draft in
                    draft.media.sourceImage = picked.encoded
                    draft.media.sourceImageName = picked.name
                    draft.media.lastExclusiveWrite = .source
                } }
            } : nil,
            addAsReference: referenceRoom ? { result in
                Task { await attachResult(result) { picked, draft in
                    draft.media.editImages.append(picked.encoded)
                    draft.media.lastExclusiveWrite = .references
                } }
            } : nil)
    }

    /// The source well exists for this recipe AND this layout draws it.
    private var canReuseAsSource: Bool {
        guard let recipe else { return false }
        return ImageConditioningWells
            .layout(recipe: recipe, model: selectedModel, media: controller.draft.media)
            .showsSourceWell
    }

    /// The strip is drawn and has room. `hasRoom` is the capability's own
    /// answer, where an absent `max_count` is UNBOUNDED.
    private var referenceRoom: Bool {
        guard let recipe else { return false }
        let layout = ImageConditioningWells
            .layout(recipe: recipe, model: selectedModel, media: controller.draft.media)
        guard let references = layout.references else { return false }
        return references.hasRoom(for: controller.draft.media.editImages.count)
    }

    private func bytes(of result: BatchResult) async -> Data? {
        guard let host, let filename = result.filename else { return nil }
        return try? await hosts.backend(for: host).media(filename, trashed: false)
    }

    private func saveResult(_ result: BatchResult) async {
        guard let data = await bytes(of: result), let filename = result.filename else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = filename
        guard await panel.begin() == .OK, let url = panel.url else { return }
        try? data.write(to: url)
    }

    private func copyResult(_ result: BatchResult) async {
        guard let data = await bytes(of: result), let image = NSImage(data: data) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([image])
    }

    /// Bytes already on a machine, base64'd off the main actor exactly as an
    /// imported file is -- the draft holds what will be sent.
    private func attachResult(
        _ result: BatchResult, apply: (ImportedPicture, inout RenderDraft) -> Void
    ) async {
        guard let data = await bytes(of: result), let filename = result.filename else { return }
        let picked = await PictureImport.encoded(data, name: filename)
        apply(picked, &controller.draft)
    }
}
