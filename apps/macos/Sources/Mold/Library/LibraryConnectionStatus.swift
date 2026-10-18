import MoldClient
import SwiftUI

/// A stable toolbar slot: connection changes never insert rows above the grid.
struct LibraryConnectionStatus: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(PrintMaterializer.self) private var materializer
    @State private var showsDetails = false
    private var offline: [MoldHost] { hosts.hosts.filter { !hosts.isUp($0) } }
    private var hasMessages: Bool { !offline.isEmpty || !hosts.failures.isEmpty || materializer.note != nil }
    var body: some View {
        Button { showsDetails.toggle() } label: {
            Label("Library connection status", systemImage: hasMessages ? "icloud.slash" : "checkmark.icloud")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
        }
        .help(hasMessages ? "Connection status and library messages" : "All machines connected")
        .accessibilityIdentifier("library-connection-status")
        .popover(isPresented: $showsDetails) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Library Status").font(.headline)
                    if library.activityDisclosure.hasDismissed {
                        Button("Show operation status") { library.activityDisclosure.restore(); showsDetails = false }
                    }
                    if library.localSaveRunning {
                        Text(library.localSaveProgress ?? "Finishing sync…")
                        Button("Stop Sync") { library.syncSession.stop(in: library) }
                            .disabled(library.localSaveStopRequested)
                    }
                    if library.bulkRunning {
                        Text(library.bulkProgress ?? "Finishing operation…")
                        Button("Stop After Current Batch") { library.bulkStopRequested = true }
                            .disabled(library.bulkStopRequested)
                    }
                    if offline.isEmpty { Text("All machines connected") }
                    else {
                        Text("Saved prints remain available while these machines are offline.")
                        ForEach(offline) { host in Label(host.name, systemImage: "icloud.slash") }
                    }
                    FailureBanner(failures: hosts.failures, dismiss: hosts.dismiss)
                    if let note = materializer.note {
                        MediaCacheNote(sentence: note) { materializer.note = nil }
                    }
                    Button("Done") { showsDetails = false }
                }.padding()
            }
            .frame(width: 360, height: 300)
        }
    }
}
