import MoldClient
import SwiftUI

/// The pinned status stays independent of machine names; full names belong
/// in the scrollable details so saved prints keep a usable viewport.
enum OfflineLibraryNotice {
    static func summary(hosts: [MoldHost]) -> String? {
        guard !hosts.isEmpty else { return nil }
        return hosts.count == 1
            ? String(localized: "Saved prints · 1 machine offline")
            : String(localized: "Saved prints · \(hosts.count) machines offline")
    }
}

/// Said at the top of the Library when it shows this device's saved copy.
struct OfflineNote: View {
    @Environment(LibraryStore.self) private var library
    @State private var showsDetails = false

    var body: some View {
        if let summary = OfflineLibraryNotice.summary(hosts: library.offlineHosts) {
            Button { showsDetails = true } label: {
                Label {
                    Text(summary).fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "icloud.slash").font(.caption).accessibilityHidden(true)
                }
                .font(.footnote)
                .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .background(Color(uiColor: .secondarySystemBackground), in: .rect(cornerRadius: 10))
            .contentShape(.rect)
            .accessibilityIdentifier("offline-library-status")
            .accessibilityHint("Shows the full list of unavailable machines.")
            .padding(.horizontal, 12)
            .padding(.top, 4)
            .sheet(isPresented: $showsDetails) {
                NavigationStack {
                    List {
                        Section {
                            Text("This device's saved prints remain available.")
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Section {
                            ForEach(library.offlineHosts) { host in
                                Text(host.name).fixedSize(horizontal: false, vertical: true)
                                    .accessibilityIdentifier("offline-library-host-\(host.id)")
                            }
                        } header: {
                            Text("Unavailable Machines").foregroundStyle(.secondaryText)
                        }
                    }
                    .accessibilityIdentifier("offline-library-details")
                    .navigationTitle("Saved Prints")
                    .navigationBarTitleDisplayMode(.inline)
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
