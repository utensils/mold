import MoldClient
import SwiftUI

/// Settings ▸ General -- this Mac's own preferences: what to do while Mold
/// is in the background, how much disk this Mac spends caching other
/// machines' prints (absorbed from the retired Storage tab, design "The pane
/// map"), and a way back to a known layout for everything on this page that
/// doesn't already have its own undo.
struct GeneralSettings: View {
    @AppStorage("badgeLandedPrints", store: AppStorageSuite.defaults)
    private var badgeLandedPrints = true
    @AppStorage("notifyRenders", store: AppStorageSuite.defaults)
    private var notifyRenders = true
    @AppStorage(PrintMaterializer.capKey, store: AppStorageSuite.defaults)
    private var capMegabytes = PrintMaterializer.defaultCapMegabytes
    @Environment(PrintMaterializer.self) private var materializer
    @State private var used = 0
    @State private var pendingReset: Destruction?

    // Internal, not private: `SettingsPanesTests` pins this against the six
    // choices `StorageSettings` used to offer, so the move didn't quietly
    // narrow or reorder them.
    static let mediaCacheChoices = [256, 512, 1_024, 2_048, 4_096, 8_192]

    var body: some View {
        Form {
            Section {
                Toggle("Badge the Dock icon with prints that arrive while Mold is in the background",
                       isOn: $badgeLandedPrints)
                Toggle("Notify when a render finishes or a job fails", isOn: $notifyRenders)
            } footer: {
                Text("Only while Mold is in the background. Coming back to the app clears the badge.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section {
                Picker("Media cache", selection: $capMegabytes) {
                    ForEach(Self.mediaCacheChoices, id: \.self) { megabytes in
                        Text(size(megabytes)).tag(megabytes)
                    }
                    Text("Off").tag(0)
                }
                LabeledContent("Using", value: used > 0 ? bytes(used) : "Nothing")
                Button("Empty Now") {
                    materializer.purge()
                    used = 0
                }
                .disabled(used == 0)
            } footer: {
                Text("""
                     Prints live on the machines that made them. Quick Look, \
                     sharing, saving and dragging one to the Finder all need a \
                     copy on this Mac, and this is how much of that Mold keeps \
                     so a second look costs nothing. It is emptied when you \
                     quit, and nothing in it is the only copy of anything.
                     """)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Section {
                Button("Reset These Preferences…") { pendingReset = resetDestruction }
            } footer: {
                Text("""
                     Puts the window layout, the remembered destination and \
                     machine, and every pane's own sort and sidebar state \
                     back to how Mold first opened. Your machines are untouched.
                     """)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .task { used = materializer.usedBytes }
        .onChange(of: capMegabytes) { _, _ in
            materializer.enforceBudget()
            used = materializer.usedBytes
        }
        .destructionDialog($pendingReset)
    }

    private var resetDestruction: Destruction {
        Destruction(
            title: "Reset These Preferences?",
            message: """
                     Window layout, remembered destination and machine, and every pane's own sort and \
                     sidebar state go back to their defaults. Your machines are untouched.
                     """,
            verb: "Reset"
        ) {
            PreferencesReset.reset(in: AppStorageSuite.defaults)
        }
    }

    /// Binary, not decimal. The cap is applied in mebibytes, so showing the
    /// 1,024 MB choice as Apple's decimal "1.07 GB" would be reporting a
    /// number nothing in the app uses.
    private func size(_ megabytes: Int) -> String {
        bytes(megabytes * 1_024 * 1_024)
    }

    private func bytes(_ count: Int) -> String {
        count.formatted(.byteCount(style: .memory))
    }
}
