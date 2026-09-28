import SwiftUI

/// Settings, as a sheet from the Machines toolbar (DESIGN.md §5.6). There is
/// deliberately no Appearance section: the system decides.
struct SettingsSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("About") {
                    AdaptiveRow {
                        Text("Version")
                    } value: {
                        Text(Self.version).monospacedDigit().textSelection(.enabled)
                    }
                    Link(destination: Self.privacyPolicy) {
                        Label("Privacy Policy", systemImage: "hand.raised")
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    static let privacyPolicy = URL(string: "https://utensils.io/mold/privacy")!

    /// `0.32.0 (1428)`: the marketing version from the workspace, then the
    /// build number, exactly as the bundle carries them.
    static var version: String {
        let info = Bundle.main.infoDictionary ?? [:]
        let marketing = info["CFBundleShortVersionString"] as? String ?? "?"
        let build = info["CFBundleVersion"] as? String ?? "?"
        return "\(marketing) (\(build))"
    }
}
