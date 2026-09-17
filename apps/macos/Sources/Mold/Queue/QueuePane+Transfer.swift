import MoldClient
import SwiftUI

// A held row's own two actions -- Pull-then-Retry and Move-to -- split out
// for size, `ModelsPane+Actions.swift`'s reason. `TransferStore.transfer`
// captions a `.sent` outcome onto its own `summary` itself; a `.refused` has
// already gone through the failure funnel, so neither needs anything here.
extension QueuePane {
    func moveTo(_ entry: QueueEntry, from host: MoldHost, to destination: MoldHost.ID) {
        Task {
            await transfers.transfer(entry, from: host.id, to: destination)
            await load()
        }
    }

    func pullThenRetry(_ model: String, entry: QueueEntry, host: MoldHost) {
        Task {
            await QueueHoldRow.pullThenRetry(model, entry: entry, host: host, downloads: downloads, queue: queue)
            await load()
        }
    }
}

extension View {
    /// The transfer caption, drawn below whatever a pane shows --
    /// `failureBanner`'s own shape, mirrored below instead of above.
    func transferCaption(_ message: String?) -> some View {
        VStack(spacing: 0) {
            self
            if let message {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.top, 4)
            }
        }
    }
}
