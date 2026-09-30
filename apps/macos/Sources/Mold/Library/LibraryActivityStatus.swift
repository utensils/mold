import SwiftUI

/// Progress changes redraw only this inset, preserving the Library grid and
/// its native context-menu owner while a background operation reports status.
struct LibraryActivityStatus: View {
    @Environment(LibraryStore.self) private var library

    var body: some View {
        VStack(spacing: 0) {
            if let progress = library.localSaveProgress {
                bulkStatusRow(progress) {
                    Button("Stop After Current Transfers") { library.localSaveStopRequested = true }
                        .disabled(library.localSaveStopRequested)
                }
            }
            if let progress = library.bulkProgress {
                bulkStatusRow(progress) {
                    Button(library.bulkEmptying ? "Stop After Current Machine" : "Stop After Current Batch") { library.bulkStopRequested = true }
                        .disabled(library.bulkStopRequested)
                }
            }
            if let progress = library.mutations.progress {
                bulkStatusRow(progress) { EmptyView() }
            }
            ForEach(library.bulkActivities.keys.sorted(by: { $0.uuidString < $1.uuidString }), id: \.self) { id in
                bulkStatusRow(library.bulkActivities[id] ?? "Working…") { EmptyView() }
            }
            if let result = library.bulkResult, !library.bulkRunning {
                HStack {
                    Text(result)
                    Spacer()
                    Button("Dismiss") { library.bulkResult = nil }
                }.padding(12).background(.bar)
            }
        }
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
