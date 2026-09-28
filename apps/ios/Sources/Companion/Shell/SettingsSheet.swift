import MoldClient
import SwiftUI

/// Settings, as a sheet (DESIGN.md §5.6). There is deliberately no Appearance
/// section: the system decides. Section headers are drawn in `.secondaryText`
/// -- the system header colour failed the contrast audit on white.
struct SettingsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(HostStore.self) private var hosts
    @Environment(AppRouter.self) private var router
    /// The iPad sidebar's Settings row shows this as a page, with no Done.
    var inSidebar = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    ForEach(hosts.hosts) { host in
                        HStack(spacing: 10) {
                            StatusDot(reachability: hosts.reachability(of: host))
                            Text(host.name)
                            Spacer(minLength: 8)
                            if hosts.defaultMachine == host.id {
                                Text("Default").foregroundStyle(.secondaryText)
                            }
                        }
                        .accessibilityElement(children: .combine)
                    }
                    Button("Add a Machine…") {
                        dismiss()
                        router.addMachine()
                    }
                } footer: {
                    // A footer, not a header: the first group sits right under
                    // the bar, where the accessibility audit measured any
                    // header below 4.5:1 however it was drawn (text, colour,
                    // margin and edge effect all tried); an unheaded first
                    // group is the Settings app's own pattern.
                    Text("The machines Mold Studio uses. Their details are under Machines.")
                        .foregroundStyle(.secondaryText)
                }
                SettingsSections()
                Section {
                    AdaptiveRow {
                        Text("Version")
                    } value: {
                        Text(Self.version).monospacedDigit()
                    }
                    .contextMenu {
                        Button("Copy Version") { UIPasteboard.general.string = Self.version }
                    }
                    Link(destination: Self.privacyPolicy) {
                        HStack {
                            // Primary, not the tint: blue words on a dark
                            // grouped row were borderline in the audit; the
                            // arrow says it is a link.
                            Text("Privacy Policy").foregroundStyle(.primary)
                            Spacer(minLength: 8)
                            // a11y: decorative -- the link's own words say where it goes.
                            Image(systemName: "arrow.up.forward").accessibilityHidden(true)
                        }
                    }
                    .tint(.primary)
                } header: {
                    SectionHeader(String(localized: "About"))
                }
            }
            // The audit tells the sheet's own elements from the dimmed
            // screen behind it by this.
            .accessibilityIdentifier("settings-sheet")
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(inSidebar ? .large : .inline)
            .toolbar {
                if !inSidebar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
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

/// A form section's title in a colour that passes contrast in both
/// appearances, still read as a header by VoiceOver.
struct SectionHeader: View {
    let title: String
    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title)
            .foregroundStyle(.secondaryText)
            .accessibilityAddTraits(.isHeader)
    }
}
