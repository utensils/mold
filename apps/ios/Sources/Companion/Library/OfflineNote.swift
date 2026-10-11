import MoldClient
import SwiftUI

/// The pinned status stays independent of machine names; full names belong
/// in the scrollable details so saved prints keep a usable viewport.
enum OfflineLibraryNotice {
    static func summary(hosts: [MoldHost], hasFailures: Bool = false) -> String? {
        guard !hosts.isEmpty else {
            return hasFailures ? String(localized: "Some requests couldn't finish") : nil
        }
        return hosts.count == 1
            ? String(localized: "Saved prints · 1 machine offline")
            : String(localized: "Saved prints · \(hosts.count) machines offline")
    }
}

/// Said at the top of the Library when it shows this device's saved copy.
struct OfflineNote: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Binding var showsDetails: Bool
    @State private var retrying = false

    var body: some View {
        let summary = OfflineLibraryNotice.summary(hosts: library.offlineHosts, hasFailures: !hosts.failures.isEmpty) ?? "All machines connected"
        return Group {
            Button { showsDetails = true } label: {
                Label(summary, systemImage: hosts.failures.isEmpty && library.offlineHosts.isEmpty ? "checkmark.icloud" : "icloud.slash")
                    .labelStyle(.iconOnly)
                    .frame(width: 44, height: 44)
            }
            .accessibilityIdentifier("offline-library-status")
            .accessibilityHint("Shows connection status and unavailable machines.")
            .sheet(isPresented: $showsDetails) {
                NavigationStack {
                    List {
                        Section { Text(summary) }
                        if !hosts.failures.isEmpty {
                            Section("Messages") {
                                FailureBanner(horizontalInset: 0)
                                    .listRowInsets(EdgeInsets())
                                    .listRowBackground(Color.clear)
                            }
                        }
                        Section {
                            Text("This device's saved prints remain available.")
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        if !library.offlineHosts.isEmpty {
                            Section {
                                ForEach(library.offlineHosts) { host in
                                    Text(host.name).fixedSize(horizontal: false, vertical: true)
                                        .accessibilityIdentifier("offline-library-host-\(host.id)")
                                }
                            } header: {
                                Text("Unavailable Machines").foregroundStyle(.secondaryText)
                            }
                        }
                    }
                    .accessibilityIdentifier("offline-library-details")
                    .navigationTitle("Saved Prints")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Retry") { retrying = true }
                                .disabled(retrying)
                                .accessibilityIdentifier("offline-library-retry")
                        }
                    }
                    .task(id: retrying) {
                        guard retrying else { return }
                        defer { retrying = false }
                        await hosts.refreshAll()
                        guard !Task.isCancelled else { return }
                        await library.reload()
                    }
                    .toolbarBackground(Color(uiColor: .systemBackground), for: .navigationBar)
                    .toolbarBackground(.visible, for: .navigationBar)
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        VStack {
                            Button { showsDetails = false } label: {
                                Text("Done").font(.body)
                                    .frame(maxWidth: .infinity, minHeight: 44)
                            }
                            .prominentAction()
                            .accessibilityIdentifier("offline-library-done")
                        }
                        .padding(12)
                        .background(Color(uiColor: .systemBackground))
                        .accessibilityElement(children: .contain)
                        .accessibilityIdentifier("offline-library-footer")
                    }
                }
                .presentationDetents([.large])
            }
        }
    }
}
