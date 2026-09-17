import MoldClient
import SwiftUI

/// Keeps the Generate draft across launches.
///
/// DEBOUNCED, and written off the main actor: the draft changes on every
/// keystroke, and a file write per character is a stutter nobody asked for.
/// The delay is a constructor parameter so a test pins the behaviour rather
/// than waiting out a constant.
@MainActor
@Observable
final class DraftPersistence {
    private let store: DraftStore
    private let delay: Duration
    private var task: Task<Void, Never>?
    /// Whether a restore has already been attempted. A second one would
    /// overwrite whatever has been typed since the first.
    private(set) var hasRestored = false
    /// The model the restored draft was being authored against, held until
    /// the model list arrives -- `GeneratePane+Models` adopts it through the
    /// ordinary path. `nil` once it has been adopted, or when there was no
    /// draft to restore.
    private(set) var restoredModel: String?

    init(store: DraftStore = DraftStore(), delay: Duration = .milliseconds(400)) {
        self.store = store
        self.delay = delay
    }

    /// Schedules a write. A newer change supersedes an older one outright --
    /// the file only ever needs to hold the LATEST draft, so a queue of
    /// snapshots would be work nobody reads.
    func schedule(_ descriptor: DraftDescriptor) {
        task?.cancel()
        task = Task { [store, delay] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            // Off the main actor: this touches the disk.
            await Task.detached(priority: .utility) { store.save(descriptor) }.value
        }
    }

    /// The descriptor to restore, ONCE. `nil` on a first launch, on a version
    /// this build does not write, and on a document that would not parse --
    /// all three of which are an empty pane, not an error.
    func restore() -> DraftDescriptor? {
        guard !hasRestored else { return nil }
        hasRestored = true
        let descriptor = store.load()
        restoredModel = descriptor?.model
        return descriptor
    }

    /// Forgets the model to adopt, once one has been. Without this a later
    /// model change would be undone by the next reachability tick re-adopting
    /// the restored one.
    func adoptedRestoredModel() { restoredModel = nil }

    /// Writes the pending change NOW, for a quit that will not wait for a
    /// debounce. Synchronous on purpose: there is no later.
    func flush(_ descriptor: DraftDescriptor) {
        task?.cancel()
        task = nil
        store.save(descriptor)
    }
}
