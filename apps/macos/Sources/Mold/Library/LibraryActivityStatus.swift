import SwiftUI

/// Progress changes redraw only this inset, preserving the Library grid and
/// its native context-menu owner while a background operation reports status.
struct LibraryActivityStatus: View {
    @Environment(LibraryStore.self) private var library
    @State private var presentation = ContextMenuPresentation<LibraryActivitySnapshot>()
    @State private var completionRevision = 0
    @State private var mutationID = UUID()
    private let tracking = ContextMenuTracking.shared

    private var syncID: String {
        "sync:\(library.syncActivityID):" + library.localSaveIssueKeys.values.sorted().joined(separator: "|")
            + (library.localSaveFailures.isEmpty ? "" : ":failure:" + library.localSaveFailures.sorted().joined(separator: "|"))
    }
    private var bulkResultID: String { bulkID + (library.bulkResultHasIssues ? ":issue:\(library.bulkResult ?? "")" : "") }
    private var bulkID: String { "bulk:\(library.bulkActivityID)" }
    private func dismiss(_ identity: String) -> some View {
        Button { library.activityDisclosure.dismiss(identity) } label: {
            Image(systemName: "xmark").accessibilityLabel("Dismiss status")
        }.buttonStyle(.borderless).help("Hide this status without stopping the operation")
    }

    var body: some View {
        let _ = completionRevision
        let current = LibraryActivitySnapshot(library)
        let status = presentation.resolve(current, isTracking: tracking.isTracking,
                                          isEmpty: current.isEmpty)
        return VStack(spacing: 0) {
            if let progress = status.localSaveProgress, library.activityDisclosure.shows(syncID) {
                bulkStatusRow(progress) {
                    Button("Stop Sync") { library.syncSession.stop(in: library) }
                        .disabled(status.localSaveStopRequested)
                    dismiss(syncID)
                }
            }
            if status.localSaveProgress == nil, !status.syncReport.isEmpty, library.activityDisclosure.shows(syncID) {
                HStack {
                    Image(systemName: status.syncHasIssues ? "exclamationmark.triangle" : "checkmark.circle")
                        .accessibilityLabel(status.syncHasIssues ? "Sync has issues" : "Sync complete")
                    Text(status.syncReport).lineLimit(3)
                    Spacer()
                    Button("Details") { library.localSaveAlertPresented = true }
                    dismiss(syncID)
                }.padding(12).background(.bar)
            }
            if let progress = status.bulkProgress, library.activityDisclosure.shows(bulkID) {
                bulkStatusRow(progress) {
                    Button(status.bulkEmptying ? "Stop After Current Machine" : "Stop After Current Batch") { library.bulkStopRequested = true }
                        .disabled(status.bulkStopRequested)
                    dismiss(bulkID)
                }
            }
            if let progress = status.mutationProgress, library.activityDisclosure.shows("mutation:\(mutationID)") {
                bulkStatusRow(progress) { dismiss("mutation:\(mutationID)") }
            }
            ForEach(status.activities.keys.sorted(by: { $0.uuidString < $1.uuidString }), id: \.self) { id in
                if library.activityDisclosure.shows("activity:\(id)") {
                    bulkStatusRow(status.activities[id] ?? "Working…") { dismiss("activity:\(id)") }
                }
            }
            if let result = status.bulkResult, !status.bulkRunning, library.activityDisclosure.shows(bulkResultID) {
                HStack {
                    Text(result)
                    Spacer()
                    Button("Dismiss") { library.activityDisclosure.dismiss(bulkResultID) }
                }.padding(12).background(.bar)
            }
        }
        .onReceive(tracking.didFinishTracking) { completionRevision = $0 }
        .onChange(of: status.mutationProgress == nil) { _, empty in if !empty { mutationID = UUID() } }
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
    let bulkProgress: String?
    let bulkEmptying: Bool
    let bulkStopRequested: Bool
    let mutationProgress: String?
    let activities: [UUID: String]
    let bulkResult: String?
    let bulkRunning: Bool

    var isEmpty: Bool {
        localSaveProgress == nil && syncReport.isEmpty && bulkProgress == nil && mutationProgress == nil
            && activities.isEmpty && (bulkResult == nil || bulkRunning)
    }

    init(_ library: LibraryStore) {
        localSaveProgress = library.localSaveProgress ?? (library.localSaveRunning ? "Finishing sync…" : nil)
        localSaveStopRequested = library.localSaveStopRequested
        syncReport = library.syncBannerReport
        syncHasIssues = library.syncHasUnacknowledgedIssues
        bulkProgress = library.bulkProgress
        bulkEmptying = library.bulkEmptying
        bulkStopRequested = library.bulkStopRequested
        mutationProgress = library.mutations.progress
        activities = library.bulkActivities
        bulkResult = library.bulkResult
        bulkRunning = library.bulkRunning
    }
}
