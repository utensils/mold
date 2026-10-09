import AppKit
import SwiftUI

struct LibrarySyncReportSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var library = library
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
                if !library.localSaveIssueKeys.isEmpty {
                    Toggle("Don’t show these unchanged media issues again", isOn: $library.syncIssueAcknowledgment)
                    Text("Sync keeps retrying. New or changed issues still appear; machine connection and authentication failures are always reported.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack {
                Button("Copy Details") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(
                        ([library.localSaveReport] + library.localSaveFailures).joined(separator: "\n"), forType: .string)
                }
                if !library.localSaveFailures.isEmpty {
                    Button("Reset Acknowledgments") {
                        library.syncSession.resetAcknowledgments()
                    }
                }
                Spacer()
                Button("Done", action: dismissReport)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 650)
        .frame(height: library.localSaveFailures.isEmpty ? nil : 500)
        .frame(minHeight: 180)
        .onExitCommand(perform: dismissReport)
    }

    private func dismissReport() {
        library.localSaveAlertPresented = false
        dismiss()
    }
}
