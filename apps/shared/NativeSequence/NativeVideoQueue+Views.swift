import MoldClient
import SwiftUI

/// What the video queue says that is not a row: saved restart attempts, the
/// errors and notices, per-machine paging, and the recovery sheet. Placed once
/// at the top of each pane's list.
struct NativeVideoQueueNotices: View {
    let queue: NativeVideoQueue
    @Bindable var presenter: NativeVideoQueuePresenter

    var body: some View {
        Group {
            NativeVideoRestartReceipts(controller: queue.restarts) { record in
                if queue.restarts.restorePending(record) {
                    presenter.discarding = nil
                    presenter.showingRecovery = true
                }
            }
            if let error = queue.restarts.error, queue.restarts.pending.isEmpty {
                Text(error).fixedSize(horizontal: false, vertical: true)
            }
            if queue.videos.unavailableRecoveryCount > 0 {
                Text("\(Plural.count(queue.videos.unavailableRecoveryCount, of: "saved video attempt")) belong to removed or changed connections. Restore their original machines to check them.")
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let error = queue.videos.error { Text(error).fixedSize(horizontal: false, vertical: true) }
            if let notice = queue.sequence.historyNotice { Text(notice).fixedSize(horizontal: false, vertical: true) }
            ForEach(Array(queue.videos.pagination.values).sorted { $0.owner.host.name < $1.owner.host.name },
                    id: \.owner.host.id) { page in
                if page.hasMore || page.error != nil || page.isLoading {
                    VStack(alignment: .leading, spacing: 6) {
                        if let error = page.error { Text(error).fixedSize(horizontal: false, vertical: true) }
                        if let target = queue.pageBackend(page.owner) {
                            Button(page.isLoading ? "Loading videos…" : page.error != nil
                                   ? "Retry videos on \(page.owner.host.name)" : "Load more videos on \(page.owner.host.name)") {
                                Task {
                                    await queue.videos.loadMore(owner: page.owner, backend: target,
                                                                current: { queue.pageCurrent(page.owner) })
                                }
                            }.disabled(page.isLoading).frame(minHeight: 44)
                        }
                    }
                }
            }
        }
        .sheet(isPresented: $presenter.showingRecovery, onDismiss: { queue.restarts.close() }) {
            NativeVideoRestartSheet(
                controller: queue.restarts, targets: queue.targets, backend: queue.backend, current: queue.current,
                deleting: presenter.discarding != nil,
                discarded: {
                    if let discarding = presenter.discarding, let target = queue.targetOf(discarding), queue.current(target) {
                        queue.sequence.removeDiscarded(discarding)
                    }
                    queue.changed?()
                }, changed: queue.changed)
        }
    }
}

/// What every video row carries whatever its chrome: the one-time saved-video
/// read and the Delete confirmation anchored to this exact row and owner.
struct NativeVideoQueueRowSupport: ViewModifier {
    let queue: NativeVideoQueue
    let item: NativeVideoQueueItem
    @Bindable var presenter: NativeVideoQueuePresenter

    private struct OutputLocator: Hashable { let owner: VideoJobOwner; let filename: String?; let completed: Bool }

    func body(content: Content) -> some View {
        content
            .confirmationDialog("Delete this job and its recovery data?", isPresented: Binding(
                get: { matches(presenter.deleting) },
                set: { if !$0, matches(presenter.deleting) { presenter.deleting = nil } }), titleVisibility: .visible) {
                    if let selected = presenter.deleting, matches(selected) {
                        Button("Delete recovery data", role: .destructive) { queue.confirmDelete(selected) }
                    }
                    Button("Cancel", role: .cancel) { presenter.deleting = nil }
                } message: {
                    Text("Saved Library videos remain. This job’s saved inputs and unfinished progress will be removed.")
                }
            .task(id: locator) { if locator != nil { await queue.checkOutput(item) } }
    }

    private var locator: OutputLocator? {
        guard case let .video(job) = item.source else { return nil }
        return OutputLocator(owner: job.owner, filename: job.detail?.outputFilename, completed: job.detail?.state == .completed)
    }

    private func matches(_ job: ObservedVideoJob?) -> Bool {
        guard let job, case let .video(mine) = item.source else { return false }
        return job.id == mine.id && job.owner == mine.owner
    }
}

extension View {
    func nativeVideoQueueRow(_ item: NativeVideoQueueItem, queue: NativeVideoQueue) -> some View {
        modifier(NativeVideoQueueRowSupport(queue: queue, item: item, presenter: queue.presenter))
    }
}

/// A row's details: everything the compact row leaves out, and every action
/// it offers spelt out as a full button.
struct NativeVideoQueueDetails: View {
    let queue: NativeVideoQueue
    let id: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        if let item = queue.item(id: id) {
            VStack(alignment: .leading, spacing: 12) {
                Text(item.headline).font(.title3.weight(.semibold))
                Text("On \(item.host.name)").foregroundStyle(.secondaryText)
                if let prompt = item.prompt, !prompt.isEmpty {
                    Text(prompt).fixedSize(horizontal: false, vertical: true)
                }
                Text(item.row.caption).font(.headline)
                if let progress = item.row.progress { ProgressView(value: progress) }
                Text(item.row.detail).foregroundStyle(.secondaryText)
                ForEach(Array(item.notes.enumerated()), id: \.offset) { _, note in
                    Text(note).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                Text("Job \(item.operation.prefix(8))").font(.caption).foregroundStyle(.secondaryText)
                    .textSelection(.enabled)
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) { buttons(item) }
                    VStack(alignment: .leading, spacing: 8) { buttons(item) }
                }
                .disabled(item.isBusy)
            }
            .buttonStyle(.bordered)
            .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Text("This video is no longer in the Queue.").onAppear { dismiss() }
        }
    }

    @ViewBuilder private func buttons(_ item: NativeVideoQueueItem) -> some View {
        ForEach(item.actions, id: \.self) { action in
            Button(item.title(action, stopRequested: queue.stopRequested(item)),
                   role: NativeVideoQueueItem.isDestructive(action) ? .destructive : nil) {
                if action == .dismiss || action == .openInLibrary || action == .startAgain || action == .delete { dismiss() }
                queue.perform(action, on: item)
            }
            .frame(minHeight: 44)
        }
    }
}
