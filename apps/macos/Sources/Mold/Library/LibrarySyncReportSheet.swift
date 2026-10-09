import AppKit
import SwiftUI

struct LibrarySyncReportSheet: View {
    @Environment(LibraryStore.self) private var library
    @State private var acknowledge = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Sync to This Mac").font(.title2.bold())
            Text(library.localSaveReport).fixedSize(horizontal: false, vertical: true)
            if !library.localSaveFailures.isEmpty {
                Text("Issues").font(.headline)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(library.localSaveFailures.enumerated()), id: \.offset) { _, failure in
                            Text(failure).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                Toggle("Don’t show these unchanged media issues again", isOn: $acknowledge)
                    .disabled(library.localSaveIssueKeys.isEmpty)
                    .onChange(of: acknowledge) { _, enabled in
                        if enabled { library.syncSession.acknowledge(Array(library.localSaveIssueKeys.values)) }
                        else { library.syncSession.unacknowledge(Array(library.localSaveIssueKeys.values)) }
                    }
                Text("Sync keeps retrying. New or changed issues still appear; machine connection and authentication failures are always reported.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Copy Details") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(
                        ([library.localSaveReport] + library.localSaveFailures).joined(separator: "\n"), forType: .string)
                }
                Button("Reset Acknowledgments") {
                    library.syncSession.resetAcknowledgments()
                    acknowledge = false
                }
                Spacer()
                Button("Done") { library.localSaveAlertPresented = false }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 650)
        .frame(height: library.localSaveFailures.isEmpty ? nil : 500)
        .frame(minHeight: 180)
    }
}
