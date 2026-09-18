import MoldClient
import SwiftUI

// Pressing Generate: which route the clip takes, the one retained picture a
// chain has to be given before it starts, and the probe both of those ask.
// Split from `GeneratePane.swift` past the file-size advisory.
extension GeneratePane {
    /// A clip longer than the checkpoint renders in one pass goes out as an
    /// EPHEMERAL chain job instead of a batch. The routing is resolved here
    /// because it needs the recipe, which the controller does not hold.
    func startRun() {
        guard let host else { return }
        let routing = recipe.flatMap {
            ClipRouting.resolve(recipe: $0, model: selectedModel, draft: controller.draft,
                                limits: advertisedChainLimits)
        }?.decision ?? .single()
        // The chain door redeems no reuse session, but the chain WIRE carries
        // the bytes per stage -- so the print's picture is fetched into the
        // draft's own well and the render goes out as an ordinary long clip
        // that starts from it. The authority is TAKEN before the await, so a
        // second press finds none and takes the ordinary synchronous path.
        if case .chain = routing, let authority = reuse.pending(for: controller.draft),
           let member = RetainedSourcePicture.member(
               of: authority, forHydrating: outgoingProbe(on: host)) {
            reuse.clear()
            Task { await attachThenRun(member, of: authority) }
            return
        }
        // Whatever a chain still cannot carry -- a mask, an identity photo --
        // is said, and only when something would actually have been hydrated.
        if case .chain = routing {
            reuse.warnIfTheRouteCannotCarryMedia(chained: true, outgoing: outgoingProbe(on: host))
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
        _ member: RetainedSourceMedia.Member, of authority: ReuseStore.Authority
    ) async {
        switch await RetainedSourcePicture.fetch(member, of: authority, hosts: hosts) {
        case let .refused(sentence):
            reuse.notice = sentence
        case let .picture(picture):
            RetainedSourcePicture.place(picture, named: authority.filename,
                                        in: &controller.draft)
            startRun()
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
