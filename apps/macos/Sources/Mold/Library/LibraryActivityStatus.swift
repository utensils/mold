import SwiftUI

/// Progress changes redraw only this inset, preserving the Library grid and
/// its native context-menu owner while a background operation reports status.
struct LibraryActivityStatus: View {
    @Environment(LibraryStore.self) private var library
    @State private var presentation = ContextMenuPresentation<LibraryActivitySnapshot>()
    @State private var completionRevision = 0
    private let tracking = ContextMenuTracking.shared

    var body: some View {
        let _ = completionRevision
        let current = LibraryActivitySnapshot(library)
        let status = presentation.resolve(current, isTracking: tracking.isTracking,
                                          isEmpty: current.isEmpty)
        return VStack(spacing: 0) {
            if let progress = status.localSaveProgress {
                bulkStatusRow(progress) {
                    Button("Stop Sync") {
                        library.syncSession.stop(in: library)
                        library.localSaveStopRequested = true
                    }
                        .disabled(status.localSaveStopRequested)
                }
            }
            if status.localSaveProgress == nil, !status.syncReport.isEmpty {
                HStack {
                    Image(systemName: status.syncHasIssues ? "exclamationmark.triangle" : "checkmark.circle")
                        .accessibilityLabel(status.syncHasIssues ? "Sync has issues" : "Sync complete")
                    Text(status.syncReport).lineLimit(3)
                    Spacer()
                    Button("Details") { library.localSaveAlertPresented = true }
                }.padding(12).background(.bar)
            }
            if status.syncEnabled {
                HStack {
                    if let next = status.syncNextRun {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            let minutes = max(0, Int(ceil(next.timeIntervalSince(context.date) / 60)))
                            Text("Next sync in \(minutes) \(minutes == 1 ? "minute" : "minutes")")
                        }
                    } else { Text("Sync every 5 minutes during this session") }
                    Spacer()
                    Button("Stop Sync") { library.syncSession.stop(in: library) }
                }.padding(12).background(.bar)
            }
            if let progress = status.bulkProgress {
                bulkStatusRow(progress) {
                    Button(status.bulkEmptying ? "Stop After Current Machine" : "Stop After Current Batch") { library.bulkStopRequested = true }
                        .disabled(status.bulkStopRequested)
                }
            }
            if let progress = status.mutationProgress {
                bulkStatusRow(progress) { EmptyView() }
            }
            ForEach(status.activities.keys.sorted(by: { $0.uuidString < $1.uuidString }), id: \.self) { id in
                bulkStatusRow(status.activities[id] ?? "Working…") { EmptyView() }
            }
            if let result = status.bulkResult, !status.bulkRunning {
                HStack {
                    Text(result)
                    Spacer()
                    Button("Dismiss") { library.bulkResult = nil }
                }.padding(12).background(.bar)
            }
        }
        .onReceive(tracking.didFinishTracking) { completionRevision = $0 }
    }
}

extension LibraryActivityStatus {
    func bulkStatusRow<Controls: View>(_ message: String,
                                      @ViewBuilder controls: () -> Controls) -> some View {
        HStack {
            ProgressView().controlSize(.small)
            Text(message).accessibilityAddTraits(.updatesFrequently)
            Spacer()
            controls()
        }
        .padding(12)
        .background(.bar)
        .accessibilityElement(children: .contain)
    }
}

/// Status data is independent of the Library grid's observation graph. Keeping
/// this value stable through completion also keeps the safe-area owner stable.
private struct LibraryActivitySnapshot {
    let localSaveProgress: String?
    let localSaveStopRequested: Bool
    let syncReport: String
    let syncHasIssues: Bool
    let syncEnabled: Bool
    let syncNextRun: Date?
    let bulkProgress: String?
    let bulkEmptying: Bool
    let bulkStopRequested: Bool
    let mutationProgress: String?
    let activities: [UUID: String]
    let bulkResult: String?
    let bulkRunning: Bool

    var isEmpty: Bool {
        localSaveProgress == nil && syncReport.isEmpty && !syncEnabled && bulkProgress == nil && mutationProgress == nil
            && activities.isEmpty && (bulkResult == nil || bulkRunning)
    }

    init(_ library: LibraryStore) {
        localSaveProgress = library.localSaveProgress ?? (library.localSaveRunning ? "Finishing sync…" : nil)
        localSaveStopRequested = library.localSaveStopRequested
        syncReport = library.localSaveReport
        syncHasIssues = !library.localSaveFailures.isEmpty
        syncEnabled = library.syncSession.isEnabled
        syncNextRun = library.syncSession.nextRun
        bulkProgress = library.bulkProgress
        bulkEmptying = library.bulkEmptying
        bulkStopRequested = library.bulkStopRequested
        mutationProgress = library.mutations.progress
        activities = library.bulkActivities
        bulkResult = library.bulkResult
        bulkRunning = library.bulkRunning
    }
}
