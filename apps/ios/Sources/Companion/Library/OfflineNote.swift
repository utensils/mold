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
                            Text("These machines aren't answering. You can keep browsing this device's saved prints.")
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Section("Unavailable Machines") {
                            ForEach(library.offlineHosts) { host in
                                Text(host.name).fixedSize(horizontal: false, vertical: true)
                                    .accessibilityIdentifier("offline-library-host-\(host.id)")
                            }
                        }
                    }
                    .accessibilityIdentifier("offline-library-details")
                    .navigationTitle("Saved Prints")
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showsDetails = false }
                        }
                    }
                }
                .presentationDetents([.large])
            }
        }
    }
}
