import Foundation
import MoldClient
import Synchronization

@MainActor
/// One automatic restore owns its cancellation state independently of a newer
/// explicit Library action or a replacement SwiftUI task.
final class DurationSourceRestoration {
    let owner: VideoJobOwner
    let source: VideoJobSource
    let sourceOwner: VideoLibrarySourceOwner?
    let token = UUID()
    nonisolated let cancelled = Mutex(false)
    init(owner: VideoJobOwner, source: VideoJobSource, sourceOwner: VideoLibrarySourceOwner?) {
        self.owner = owner; self.source = source; self.sourceOwner = sourceOwner
    }
    nonisolated func cancel() { cancelled.withLock { $0 = true } }
    nonisolated var isCancelled: Bool { cancelled.withLock { $0 } }
}

