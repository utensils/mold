import ActivityKit
import Foundation
import ImageIO
import MoldClient
import UIKit
import UniformTypeIdentifiers

/// The render Live Activity, driven by the same `RunState` the canvas draws
/// (DESIGN.md §5.7): started when a render is admitted, updated about once a
/// second, marked stale on the way to the background, and ended 15 minutes
/// after it finishes. Also where a render that settles while the app is away
/// becomes a notification, and a finished one is saved to Photos when asked.
@Observable
final class ActivityCoordinator {
    @ObservationIgnored let generate: GenerateController
    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let notifier: Notifier
    @ObservationIgnored let library: LibraryStore
    @ObservationIgnored private var current: ActiveBatch?
    @ObservationIgnored private var lastPush = Date.distantPast
    @ObservationIgnored private var pending: Task<Void, Never>?
    @ObservationIgnored private var rate = StepRate()
    @ObservationIgnored private var saver: PrintActions

    init(generate: GenerateController, hosts: HostStore, notifier: Notifier, library: LibraryStore) {
        self.generate = generate
        self.hosts = hosts
        self.notifier = notifier
        self.library = library
        saver = PrintActions(hosts: hosts)
        StopRenderIntent.handler = { [weak self] id in await self?.stop(clientBatchId: id) }
        generate.settled = { [weak self] batch, run in self?.terminal(batch, run) }
        observe()
        #if DEBUG
        LiveActivityFixture.startIfRequested()
        #endif
    }

    private func observe() {
        withObservationTracking {
            _ = generate.run
        } onChange: { [weak self] in
            Task { @MainActor in
                self?.changed()
                self?.observe()
            }
        }
    }

    private var enabled: Bool {
        Preference.isOn(Preference.liveActivities) && ActivityAuthorizationInfo().areActivitiesEnabled
    }

    /// Running updates, coalesced to about one a second. Where a batch ENDS
    /// is `terminal`, told synchronously by the controller with that batch --
    /// by the time an observation fires the next queued batch may already be
    /// on screen.
    private func changed() {
        switch generate.run {
        case .running:
            current = generate.activeBatch
            let now = Date.now
            if now.timeIntervalSince(lastPush) >= 1 { push() } else if pending == nil {
                pending = Task { [weak self] in
                    try? await Task.sleep(for: .seconds(1))
                    self?.pending = nil
                    self?.push()
                }
            }
        case .submitting:
            // The first render is when asking makes sense; the system asks once.
            Task { await notifier.requestAuthorization() }
        case .idle, .finished, .failed:
            break
        }
    }

    private func terminal(_ batch: ActiveBatch, _ run: RunState) {
        pending?.cancel()
        pending = nil
        rate.reset()
        if current?.clientBatchId == batch.clientBatchId { current = nil }
        switch run {
        case let .finished(outcome, host):
            settle(batch, outcome: outcome, host: host)
        case let .failed(reason):
            end(batch.clientBatchId, with: finalState(run, batch))
            if UIApplication.shared.applicationState != .active {
                notifier.post(.failed(reason), batch: batch, machine: machine(batch), print: nil)
            }
        default:
            // Stopped from here: the activity goes at once.
            end(batch.clientBatchId, with: nil)
        }
    }

    private func machine(_ batch: ActiveBatch) -> String { hosts.host(batch.host)?.name ?? "" }

    private func finalState(_ run: RunState, _ batch: ActiveBatch) -> GenerationActivityAttributes.ContentState? {
        ActivityProjection.state(for: run, machine: machine(batch), waiting: generate.queued.count,
                                 remaining: nil, preview: previewName(batch))
    }

    private func settle(_ batch: ActiveBatch, outcome: BatchOutcome, host: MoldHost.ID) {
        end(batch.clientBatchId, with: finalState(.finished(outcome, host: host), batch))
        let first = outcome.results.first?.filename.map { PrintID(host: host, filename: $0) }
        if UIApplication.shared.applicationState != .active {
            notifier.post(.finished(count: outcome.results.count), batch: batch, machine: machine(batch),
                          print: first)
        }
        if Preference.isOn(Preference.autoSaveToPhotos) {
            let names = outcome.results.compactMap(\.filename)
            Task { await autoSave(names, on: host) }
        }
    }

    /// The finished prints into Photos, once the Library lists them.
    private func autoSave(_ names: [String], on host: MoldHost.ID) async {
        var found: [LibraryEntry] = []
        for _ in 0..<10 {
            found = names.compactMap { name in
                library.pool.first { $0.everyCopy.contains { $0.hostID == host && $0.print.filename == name } }
            }
            if found.count == names.count { break }
            await library.reload(host)
            try? await Task.sleep(for: .milliseconds(500))
        }
        if !found.isEmpty { saver.saveToPhotos(found) }
    }

    // MARK: - ActivityKit

    private func push() {
        lastPush = .now
        guard enabled, let batch = current, case let .running(_, progress) = generate.run else { return }
        writePreview(progress?.previewData, for: batch)
        let remaining = rate.remaining(progress)
        guard let state = ActivityProjection.state(for: generate.run, machine: machine(batch),
                                                   waiting: generate.queued.count, remaining: remaining,
                                                   preview: previewName(batch)) else { return }
        let content = ActivityContent(state: state, staleDate: ActivityProjection.staleDate(for: state))
        if activity(for: batch.clientBatchId) != nil {
            let id = batch.clientBatchId
            Task { await Self.update(id, content) }
        } else {
            let attributes = GenerationActivityAttributes(prompt: batch.prompt, machine: machine(batch),
                                                          clientBatchId: batch.clientBatchId,
                                                          host: batch.host.uuidString)
            // Refused on iPad and when the person turned them off: not an error.
            _ = try? Activity.request(attributes: attributes, content: content, pushType: nil)
        }
    }

    private func activity(for clientBatchId: String) -> Activity<GenerationActivityAttributes>? {
        Activity<GenerationActivityAttributes>.activities.first { $0.attributes.clientBatchId == clientBatchId }
    }

    // `Activity` is not Sendable: each async change looks it up again on the
    // side that awaits, rather than carrying one across.
    nonisolated private static func find(_ id: String) -> Activity<GenerationActivityAttributes>? {
        Activity<GenerationActivityAttributes>.activities.first { $0.attributes.clientBatchId == id }
    }

    nonisolated private static func update(_ id: String, _ content: ActivityContent<GenerationActivityAttributes.ContentState>) async {
        await find(id)?.update(content)
    }

    nonisolated private static func end(_ id: String, _ content: ActivityContent<GenerationActivityAttributes.ContentState>?,
                                        after: Date?) async {
        await find(id)?.end(content, dismissalPolicy: after.map { .after($0) } ?? .immediate)
    }

    /// Ends it: with a final state that stays 15 minutes, or at once.
    func end(_ clientBatchId: String, with state: GenerationActivityAttributes.ContentState?) {
        guard activity(for: clientBatchId) != nil else { return }
        let content = state.map { ActivityContent(state: $0, staleDate: nil) }
        let after = state == nil ? nil : Date.now.addingTimeInterval(ActivityProjection.dismissAfter)
        Task { await Self.end(clientBatchId, content, after: after) }
    }

    /// On the way to the background: one last update, so the stale date is
    /// set from the latest estimate.
    func enteredBackground() { push() }

    /// From background refresh: a batch still going, as its machine reports it.
    func reflect(_ batch: ActiveBatch, status: BatchStatus, progress: JobProgress?) {
        guard enabled, activity(for: batch.clientBatchId) != nil,
              let state = ActivityProjection.state(for: .running(status, progress), machine: machine(batch),
                                                   waiting: 0, remaining: nil, preview: previewName(batch))
        else { return }
        let id = batch.clientBatchId
        let content = ActivityContent(state: state, staleDate: ActivityProjection.staleDate(for: state))
        Task { await Self.update(id, content) }
    }

    /// The Stop button on the Lock Screen: the render on its machine, and the
    /// activity with it.
    func stop(clientBatchId id: String) async {
        if generate.activeBatch?.clientBatchId == id {
            generate.stop()
        } else if let batch = generate.ledger.batches.first(where: { $0.clientBatchId == id }),
                  let host = hosts.host(batch.host) {
            try? await hosts.backend(for: host).cancelBatch(id: batch.id)
            generate.ledger.remove(id)
        }
        end(id, with: nil)
    }

    // MARK: - Preview files

    private func previewName(_ batch: ActiveBatch) -> String { "\(batch.clientBatchId).jpg" }
    private func previewURL(_ batch: ActiveBatch) -> URL {
        AppGroup.activityPreviews.appending(path: previewName(batch))
    }

    /// Downsampled to 256 px: ActivityKit draws small, and the file is read
    /// by another process.
    private func writePreview(_ data: Data?, for batch: ActiveBatch) {
        guard let data, let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceThumbnailMaxPixelSize: 256,
              ] as CFDictionary),
              let out = CGImageDestinationCreateWithURL(previewURL(batch) as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(out, image, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        CGImageDestinationFinalize(out)
    }
}
