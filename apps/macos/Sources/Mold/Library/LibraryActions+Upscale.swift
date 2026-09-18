import MoldClient
import SwiftUI

// Make Bigger…, and what it is offered for.
//
// Its own file rather than more of `LibraryActions`, which is already past
// the type-size budget: the decision and the call both live here, and the
// store that does the work is `UpscaleStore`.
@MainActor
extension LibraryActions {

    /// Whether this exact selection can be made bigger.
    ///
    /// ONE print: the clip half is a durable job per print, and a selection
    /// of forty would queue a machine full of work from one click. The rest
    /// of the answer is the machine's own -- `UpscaleStore.canUpscale` asks
    /// the capability block, and absence there means the item is ABSENT.
    ///
    /// `upscales == nil` is a context with no store to act through (a
    /// preview, the viewer built without one), which is also no.
    func canUpscale(_ targets: [LibraryEntry]) -> Bool {
        guard let upscales, targets.count == 1, let entry = targets.first else { return false }
        // Absent while this app is already making THIS print bigger, so the
        // action cannot be pressed twice from a menu at all. Where it got to
        // is the Queue pane's Also Running row.
        return upscales.canUpscale(entry) && !upscales.isBusy(with: entry)
    }

    /// Makes this print bigger on the machine that holds it.
    ///
    /// Nothing is awaited here: a clip upscale is a durable job on that
    /// machine and a still is a render that takes as long as it takes, and
    /// neither is something a menu should hold a window open for. Where it
    /// got to is `UpscaleStore`'s to say -- in the Queue pane, under work
    /// with no queue row of its own.
    func upscale(_ targets: [LibraryEntry], using model: String? = nil) {
        guard let upscales, let entry = targets.first else { return }
        Task { await upscales.start(entry, model: model) }
    }

    /// The installed upscalers to offer for this selection, the default
    /// first. Cache-only -- see `UpscaleStore.upscalerOptions`.
    func upscalerOptions(for targets: [LibraryEntry]) -> [UpscalerOption] {
        guard let upscales, let entry = targets.first else { return [] }
        return upscales.upscalerOptions(on: entry.hostID)
    }
}
