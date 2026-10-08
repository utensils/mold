import MoldClient
import SwiftUI

// Pressing Generate: which route the clip takes, the one retained picture a
// chain has to be given before it starts, and the probe both of those ask.
// Split from `GeneratePane.swift` past the file-size advisory.
extension GeneratePane {
    /// A clip longer than the checkpoint renders in one pass goes out as an
    /// EPHEMERAL chain job instead of a batch. The routing is resolved here
    /// because it needs the recipe, which the controller does not hold.
    func startRun() { startRun(accepted: []) }

    /// `accepted` is the licence ids accepted since the placement answer
    /// was read, so a retry after the sheet does not ask for them again.
    /// `licenceSettled` skips the licence gate once the machine could not
    /// answer the fresh probe it asked for (`GeneratePane+Licence.swift`).
    func startRun(accepted: Set<String>, licenceSettled: Bool = false) {
        if let refusal = drafts.recoveryRefusal {
            controller.submissionFeedback.begin(refusal, phase: .refused)
            return
        }
        guard let host else {
            controller.submissionFeedback.begin("Choose a connected machine before generating.", phase: .refused)
            return
        }
        guard controller.modelName != nil else {
            controller.submissionFeedback.begin("Choose a model before generating.", phase: .refused)
            return
        }
        if let refusal = reuse.referenceRefusal(for: controller.draft) {
            controller.submissionFeedback.begin(refusal, phase: .refused)
            reuse.notice = refusal
            return
        }
        // A render that would FETCH a gated model -- Qwen Image 2.1 and its
        // turbo tiers, Qwen Research -- asks for the terms before anything is
        // queued, the way the web does (`licenseRequirements`); accepting
        // runs this press again (`GeneratePane+Licence.swift`).
        if !licenceSettled, holdsForLicence(on: host, accepted: accepted) { return }
        let routing = recipe.flatMap {
            ClipRouting.resolve(recipe: $0, model: selectedModel, draft: controller.draft,
                                limits: advertisedChainLimits)
        }?.decision ?? .single()
        // The chain door redeems no reuse session, but the chain WIRE carries
        // the bytes per stage -- so the print's picture is fetched into the
        // draft's own well and the render goes out as an ordinary long clip
        // that starts from it. Hold additional presses while restoring the
        // source; keep the archive locator and fence late placement on edits.
        if case .chain = routing, let authority = reuse.pending(for: controller.draft),
           RetainedSourcePicture.member(of: authority, forHydrating: outgoingProbe(on: host)) != nil {
            let draft = controller.draft
            let outgoing = outgoingProbe(on: host)
            let fence = reuse.beginSourceSubmission()
            let feedbackID = controller.submissionFeedback.begin("Restoring the source media before sending…")
            Task { await attachThenRun(draft, outgoing: outgoing, fence: fence, accepted: accepted, feedbackID: feedbackID) }
            return
        }
        // Whatever a chain still cannot carry -- a mask, an identity photo --
        // is said, and only when something would actually have been hydrated.
        if case .chain = routing {
            reuse.warnIfTheRouteCannotCarryMedia(chained: true, outgoing: outgoingProbe(on: host))
        }
        if let model = controller.modelName {
            promptHistory.remember(controller.draft.prompt, model: model, on: host.id)
        }
        // TAKEN, not read: a handle is good for one admission and a relay's
        // bytes ride the request that took them, so the submit that gets this
        // is the last one to have it. That is what stops a print conditioning
        // renders nobody asked for, and what stops a print the machine can no
        // longer honour refusing every render after the first.
        controller.submit(
            on: host, backend: hosts.backend(for: host), routing: routing,
            retained: reuse.take(for: controller.draft).map {
                RetainedMediaHydration(authority: $0, hosts: hosts)
            })
    }

    /// Puts the print's picture in the source well, then runs -- or says why
    /// it could not, and runs nothing. A press that quietly rendered a long
    /// clip without the picture it was supposed to start from is the thing
    /// this whole path exists to stop.
    func attachThenRun(
        _ draft: RenderDraft, outgoing: GenerateRequest?, fence: Int,
        accepted: Set<String> = [], feedbackID: UUID? = nil
    ) async {
        if let placed = await reuse.sourceForSubmission(in: draft, outgoing: outgoing,
            live: { controller.draft }, fence: fence, mediaRevisions: { controller.mediaRevisions }) {
            controller.draft = placed
            startRun(accepted: accepted)
        } else if let feedbackID {
            controller.submissionFeedback.update(
                reuse.notice ?? "The source media or draft changed. Check the source and press Generate again.",
                phase: .refused, for: feedbackID)
        }
    }

    func outgoingProbe(on host: MoldHost) -> GenerateRequest? {
        RetainedSourcePicture.outgoing(controller, on: host, hosts: hosts)
    }

    func cancelRun() { controller.stop() }

    func refreshPlacement() {
        host.map(controller.refreshPlacement(on:))
    }
}
