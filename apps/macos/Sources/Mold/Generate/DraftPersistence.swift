import AppKit
import Combine
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
    @ObservationIgnored private var termination: AnyCancellable?
    @ObservationIgnored private var pending: (descriptor: DraftDescriptor, inputs: DraftInputSnapshot?)?
    /// Whether a restore has already been attempted. A second one would
    /// overwrite whatever has been typed since the first.
    private(set) var hasRestored = false
    /// The model the restored draft was being authored against, held until
    /// the model list arrives -- `GeneratePane+Models` adopts it through the
    /// ordinary path. `nil` once it has been adopted, or when there was no
    /// draft to restore.
    private(set) var restoredModel: String?
    private(set) var restoredInputs: DraftInputSnapshot?
    private(set) var saveNotice: String?
    private(set) var recoveryRefusal: String?
    private var failedInputDigest: String?

    init(store: DraftStore = DraftStore(), delay: Duration = .milliseconds(400), notifications: NotificationCenter = .default) {
        self.store = store
        self.delay = delay
        // AppStores owns this object after Generate disappears. Delivery is
        // synchronous: a queued Task would not run before process termination.
        termination = notifications.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self, let pending = self.pending else { return }
                    self.flush(pending.descriptor, inputs: pending.inputs)
                }
            }
    }

    /// Schedules a write. A newer change supersedes an older one outright --
    /// the file only ever needs to hold the LATEST draft, so a queue of
    /// snapshots would be work nobody reads.
    func schedule(_ descriptor: DraftDescriptor, inputs: DraftInputSnapshot? = nil) {
        enqueue(descriptor, inputs: inputs, after: delay)
    }

    /// Leaving Generate must not wait for an in-progress input encoding/write.
    /// Save immediately in the background; the app-owned persistence object
    /// outlives the pane. Final application termination still calls flush.
    func saveOnNavigation(_ descriptor: DraftDescriptor, inputs: DraftInputSnapshot? = nil) {
        enqueue(descriptor, inputs: inputs, after: .zero)
    }

    private func enqueue(_ descriptor: DraftDescriptor, inputs: DraftInputSnapshot?, after delay: Duration) {
        pending = (descriptor, inputs)
        task?.cancel()
        let revision = store.reserveWrite()
        let (descriptor, inputs) = preservingUnreadableInputs(descriptor, inputs: inputs)
        task = Task { [store, delay] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            // Off the main actor: this touches the disk.
            let saved = await Task.detached(priority: .utility) {
                store.save(descriptor, inputs: inputs, revision: revision)
            }.value
            guard !Task.isCancelled else { return }
            self.saveNotice = saved ? self.recoveryRefusal : "This draft could not be saved. Keep Mold open; check disk space and the size of attached files before quitting."
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
        if let descriptor {
            do { restoredInputs = try store.loadInputs(for: descriptor) }
            catch {
                failedInputDigest = descriptor.localInputsSHA256
                recoveryRefusal = "Some saved input files could not be restored. Reattach any files you need, then choose Use current inputs to save this draft."
                saveNotice = recoveryRefusal
            }
        }
        return descriptor
    }

    /// Forgets the model to adopt, once one has been. Without this a later
    /// model change would be undone by the next reachability tick re-adopting
    /// the restored one.
    func adoptedRestoredModel() { restoredModel = nil }

    /// Writes the pending change NOW, for a quit that will not wait for a
    /// debounce. Synchronous on purpose: there is no later.
    func flush(_ descriptor: DraftDescriptor, inputs: DraftInputSnapshot? = nil) {
        // The live pane may supply a newer edit before its onChange fires.
        // Whichever termination observer runs last must retain that snapshot.
        pending = (descriptor, inputs)
        task?.cancel()
        task = nil
        let (descriptor, inputs) = preservingUnreadableInputs(descriptor, inputs: inputs)
        let saved = store.save(descriptor, inputs: inputs)
        saveNotice = saved ? recoveryRefusal : "This draft could not be saved. Keep Mold open; check disk space and the size of attached files before quitting."
    }

    func discardUnavailableInputs() {
        failedInputDigest = nil
        recoveryRefusal = nil
        saveNotice = nil
    }

    private func preservingUnreadableInputs(_ descriptor: DraftDescriptor, inputs: DraftInputSnapshot?)
        -> (DraftDescriptor, DraftInputSnapshot?) {
        guard let failedInputDigest else { return (descriptor, inputs) }
        var preserved = descriptor
        preserved.localInputsSHA256 = failedInputDigest
        return (preserved, nil)
    }
}
