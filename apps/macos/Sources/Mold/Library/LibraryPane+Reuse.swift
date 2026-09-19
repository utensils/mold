import MoldClient
import SwiftUI

// Reusing a Library print in Generate, including the retained-media probe.
extension LibraryPane {
    /// Seeds the Generate pane from a finished print and goes there.
    ///
    /// The model is adopted from the machine that MADE the print, because a
    /// model installed on one host is not available on another.
    func reuse(_ entry: LibraryEntry) {
        let metadata = entry.print.metadata
        // Keep the current contract before replacing the draft. A print may
        // name no model, or name one that has since been removed; in either
        // case the selected Generate model remains the request authority.
        let currentModel = selectedGenerateModel()
        generate.draft = RenderDraft(reusing: metadata)
        let fence = reuseStore.begin()
        var adoptedPrintModel = false
        if let name = metadata.model {
            if let model = models.model(named: name, on: entry.hostID) {
                generate.adopt(model: model, on: entry.hostID, keepingDraft: true)
                adoptedPrintModel = true
            } else {
                // The rest of the recipe still restores -- the numbers, the
                // filing, the sampler -- and the style chip keeps the name
                // the print was made with. Saying so is the whole fix: the
                // controls would otherwise be reconciled against nothing and
                // silently describe a model that is not there.
                reuseStore.notice = "\(name) isn\u{2019}t on \(entry.hostName) any more. "
                    + "Everything else about this print is restored."
            }
        }
        if !adoptedPrintModel, let currentModel {
            generate.reconcileReusedDraft(with: currentModel)
        }
        destination = .generate
        // AFTER the adopt, which clamps, parks and echoes the pipeline: this
        // is the draft the pane will show, and the authority is good only
        // while the draft still IS it.
        reuseStore.arm(generate.draft)
        // ALWAYS ask, on every machine that lists this print. The server is
        // the only authority on what it retained -- inline source video,
        // audio and mask bytes leave no marker in the metadata at all -- and
        // mirroring an output does not copy the producing machine's private
        // archive, so one copy's blank says nothing about another's.
        let copies = library.items
            .filter { $0.print.filename == entry.print.filename }
            .map(\.id)
        let ordered = [entry.id] + copies.filter { $0 != entry.id }
        Task {
            await reuseStore.probe(ordered, fence: fence, disclosing: metadata)
            // Then the picture itself, into the well, so the person can see
            // what the render starts from and set its strength.
            let outgoing = hosts.host(entry.hostID).flatMap {
                RetainedSourcePicture.outgoing(generate, on: $0, hosts: hosts)
            }
            if let placed = await reuseStore.placePicture(
                in: generate.draft, outgoing: outgoing, live: { generate.draft }) {
                generate.draft = placed
            }
        }
    }

    /// Resolves the same selected model as GeneratePane: explicit machine,
    /// then adoption host, then Auto's current preferred host.
    private func selectedGenerateModel() -> Model? {
        guard let name = generate.modelName,
              let hostID = generate.machineChoice ?? generate.hostID ?? hosts.preferredHost?.id
        else { return nil }
        return models.model(named: name, on: hostID)
    }
}
