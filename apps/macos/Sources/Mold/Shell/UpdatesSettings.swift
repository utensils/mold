import SwiftUI

/// Settings ▸ General ▸ Updates.
///
/// The section remains visible in builds where the updater is deliberately
/// gated. That makes the release feature discoverable while explaining why a
/// Debug, UAT or test build must not check a feed or replace itself.
struct UpdatesSettings: View {
    var body: some View {
        if let updates = SoftwareUpdates.shared {
            UpdatesGroup(updates: updates)
        } else {
            UnavailableUpdatesGroup()
        }
    }
}

private struct UnavailableUpdatesGroup: View {
    private var channel: UpdateChannel {
        UpdateChannel(
            stored: AppStorageSuite.defaults.string(forKey: UpdateChannel.storageKey))
    }

    var body: some View {
        Section {
            Picker("Channel", selection: .constant(channel)) {
                ForEach(UpdateChannel.allCases) { choice in
                    Text(choice.label).tag(choice)
                }
            }
            .pickerStyle(.segmented)
            .disabled(true)
            Button("Check Now") {}
                .disabled(true)
        } header: {
            Text("Updates")
        } footer: {
            Text(
                """
                Updates are unavailable in this build. Install and open a \
                signed Release build to choose a channel, check for updates \
                and install them.
                """)
                .font(.caption)
                .foregroundStyle(.secondary)
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
            Picker("Channel", selection: $channel) {
                ForEach(UpdateChannel.allCases) { choice in
                    Text(choice.label).tag(choice)
                }
            }
            .pickerStyle(.segmented)
            // The feed for an interactive check has already been chosen.
            // Keep the visible selection truthful until that check settles.
            .disabled(!updates.canCheckForUpdates)
            Button("Check Now") { updates.checkForUpdates() }
                .disabled(!updates.canCheckForUpdates)
            Toggle("Automatically check for updates", isOn: $automaticallyChecks)
            Toggle("Automatically download updates", isOn: $automaticallyDownloads)
                // Sparkle's scheduler only downloads on a check it made, so
                // this means nothing on its own.
                .disabled(!automaticallyChecks)
            LabeledContent("Last checked", value: lastCheckedDescription)
        } header: {
            Text("Updates")
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
