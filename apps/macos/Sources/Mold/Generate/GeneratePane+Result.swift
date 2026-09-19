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
                Task { await attachResult(result, as: .source) }
            } : nil,
            addAsReference: referenceRoom ? { result in
                Task { await attachResult(result, as: .reference) }
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

    /// The machine the result is ON. Never `host`: that is the pane's current
    /// choice, which may have moved since Generate was pressed.
    private var finishedHost: MoldHost? { controller.run.finishedHost.flatMap(hosts.host) }

    /// Reports rather than shrugging: Save a Copy and Copy both start with
    /// this fetch, and a machine that has gone away or refused made all three
    /// of their buttons do nothing at all.
    private func bytes(of result: BatchResult) async -> Data? {
        guard let host = finishedHost, let filename = result.filename else { return nil }
        do {
            return try await hosts.backend(for: host).media(filename, trashed: false)
        } catch {
            hosts.report(error, on: host.id, doing: "fetch that picture")
            return nil
        }
    }

    private func saveResult(_ result: BatchResult) async {
        guard let data = await bytes(of: result), let filename = result.filename else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = filename
        guard await panel.begin() == .OK, let url = panel.url else { return }
        do {
            try data.write(to: url)
        } catch {
            guard let host = finishedHost else { return }
            hosts.report(error, on: host.id, doing: "save that picture")
        }
    }

    private func copyResult(_ result: BatchResult) async {
        guard let data = await bytes(of: result), let image = NSImage(data: data) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([image])
    }

    /// Bytes already on a machine, base64'd off the main actor exactly as an
    /// imported file is -- the draft holds what will be sent.
    private func attachResult(_ result: BatchResult, as kind: LibraryAttachmentKind) async {
        let fence = DraftAttachmentFence(controller: controller, reuse: reuse)
        guard let data = await bytes(of: result), let filename = result.filename else { return }
        guard fence.permits(controller, reuse: reuse), let recipe else { return }
        let picked: ImportedPicture
        do {
            picked = try await PictureImport.conforming(
                data, name: filename, accepting: PictureImport.engineReadable)
        } catch {
            guard fence.permits(controller, reuse: reuse),
                  let host = finishedHost else { return }
            hosts.report(error, on: host.id, doing: "use that picture")
            return
        }
        guard fence.permits(controller, reuse: reuse) else { return }
        var draft = controller.draft
        switch kind {
        case .source:
            guard canReuseAsSource else { return }
            DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: recipe)
        case .reference:
            let layout = ImageConditioningWells.layout(
                recipe: recipe, model: selectedModel, media: draft.media)
            guard let capability = layout.references,
                  capability.hasRoom(for: draft.media.editImages.count) else { return }
            DraftPictureAttachment.addReference(picked, to: &draft, capability: capability)
        }
        controller.draft = draft
        reuse.clear()
    }
}
