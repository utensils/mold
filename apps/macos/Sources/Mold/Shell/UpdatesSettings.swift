import SwiftUI

/// Settings ▸ General ▸ Updates.
///
/// Absent, not disabled, in a build with no updater (`SoftwareUpdates.shared`
/// is `nil` in Debug, under the UAT suite and in the test host), so this is a
/// `Section` that simply does not appear rather than one full of dead
/// controls.
struct UpdatesSettings: View {
    var body: some View {
        if let updates = SoftwareUpdates.shared {
            UpdatesGroup(updates: updates)
        }
    }
}

private struct UpdatesGroup: View {
    let updates: SoftwareUpdates

    /// Local state, seeded once and written back only on a user change.
    /// Sparkle's own preferences recipe is explicit about this -- its
    /// properties are backed by the host bundle's user defaults and must be
    /// SET only when the user changes them, never re-asserted on every render
    /// (https://sparkle-project.org/documentation/preferences-ui).
    @State private var channel: UpdateChannel
    @State private var automaticallyChecks: Bool
    @State private var automaticallyDownloads: Bool
    @State private var lastChecked: Date?

    init(updates: SoftwareUpdates) {
        self.updates = updates
        _channel = State(initialValue: updates.channel)
        _automaticallyChecks = State(initialValue: updates.automaticallyChecksForUpdates)
        _automaticallyDownloads = State(initialValue: updates.automaticallyDownloadsUpdates)
        _lastChecked = State(initialValue: updates.lastUpdateCheckDate)
    }

    var body: some View {
        Section {
            Picker("Updates", selection: $channel) {
                ForEach(UpdateChannel.allCases) { choice in
                    Text(choice.label).tag(choice)
                }
            }
            .pickerStyle(.segmented)
            Toggle("Automatically check for updates", isOn: $automaticallyChecks)
            Toggle("Automatically download updates", isOn: $automaticallyDownloads)
                // Sparkle's scheduler only downloads on a check it made, so
                // this means nothing on its own.
                .disabled(!automaticallyChecks)
            LabeledContent("Last checked", value: lastCheckedDescription)
        } footer: {
            Text("""
                 Stable updates arrive when a release is published. Nightly \
                 builds come from every change merged to main — newer, far \
                 less tested, and sometimes replaced more than once a day. \
                 Moving back to Stable waits for a release newer than the \
                 nightly you are on.
                 """)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .onChange(of: channel) { _, choice in updates.channel = choice }
        .onChange(of: automaticallyChecks) { _, isOn in
            updates.automaticallyChecksForUpdates = isOn
        }
        .onChange(of: automaticallyDownloads) { _, isOn in
            updates.automaticallyDownloadsUpdates = isOn
        }
        // The date moves while this pane is open -- a check started from the
        // menu lands here without a reopen.
        .onChange(of: updates.canCheckForUpdates) { _, _ in
            lastChecked = updates.lastUpdateCheckDate
        }
    }

    private var lastCheckedDescription: String {
        guard let lastChecked else { return "Never" }
        return lastChecked.formatted(date: .abbreviated, time: .shortened)
    }
}
