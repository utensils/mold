import MoldClient
import SwiftUI

/// One `SettingKey` drawn against a machine's row: `LabeledContent` with the
/// label, the editor, the help as a `.caption` footnote, and the refusal
/// underneath in the machine's own words. `.choice` -> `Picker`, `.number`
/// -> a `TextField` plus a `Stepper` where the key declares a step,
/// `.toggle` -> `Toggle`, `.text` -> `TextField` -- all in
/// `SettingRow+Editors`. Text and number commits share `ConfigValueField`'s
/// blur rule (`commitScalar`) rather than a second copy of it -- a curated
/// key's own `.editor` differs from `ConfigEntry`'s inferred one only for
/// `.choice`, and a choice commits immediately on selection, the same way
/// `ConfigValueField`'s `.toggle` already does.
struct SettingRow: View {
    let setting: SettingKey
    let entry: ConfigEntry
    let refusal: String?
    let onSet: (ConfigScalar) async -> Void

    // Internal, not private: `SettingRow+Editors` reads these, and `private`
    // does not cross a file boundary even within one type -- the same reason
    // `HostEditor`'s own fields are internal.
    @State var text = ""
    @FocusState var focused: Bool

    /// What one row draws, or nothing -- pure, the same `DiscoverRow.resolve`
    /// idiom `ConfigValueField.Plan` already uses. `entry == nil` is a key
    /// this build curates that the machine's own listing never mentioned --
    /// an older host, or `umt5_variant`'s answer if it were ever curated.
    struct Plan: Equatable {
        let setting: SettingKey
        let entry: ConfigEntry
        let refusal: String?

        /// The machine was started with this key in its environment, so a PUT
        /// answers 403 `ENV_OVERRIDDEN` before it tries
        /// (`routes_config.rs:195-201`). `ConfigValueField` has always drawn
        /// such a row read-only; a curated pane drew a live stepper that
        /// snapped back the moment it was dragged (review 05-M11).
        var isEnvOwned: Bool { entry.isEnvOwned }

        /// Whether the number editor draws a `Stepper` beside its field --
        /// only where the key itself declares a step, and never on a row
        /// nothing can write.
        var showsStepper: Bool {
            guard !isEnvOwned, case let .number(_, _, step) = setting.editor else { return false }
            return step != nil
        }
    }

    static func resolve(_ setting: SettingKey, entry: ConfigEntry?, refusal: String? = nil) -> Plan? {
        guard let entry else { return nil }
        return Plan(setting: setting, entry: entry, refusal: refusal)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            LabeledContent(setting.label) {
                editor
                    .help(setting.help)
                    .accessibilityLabel(setting.label)
            }
            Text(setting.help)
                .font(.caption)
                .foregroundStyle(.secondary)
            if entry.isEnvOwned {
                Text(Self.envOwnedReason(entry))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            // The server's own answer (`entry.needsRestart`), never a
            // client-authored list -- `SourceBadge` draws the identical
            // caption in Advanced, but a curated pane shows no source badge
            // for this to ride beside, so it stands alone here.
            if entry.needsRestart {
                Text("Needs a restart")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if let refusal {
                Text(refusal)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .task(id: entry) { text = entry.editableText }
        // The same list Advanced's own rows carry, minus Reset: a curated
        // pane has no reset control to mirror, and offering one only here
        // would make the two panes disagree about what a row can do.
        .rowActionMenu(ConfigRowActions.offered(for: entry).filter { $0.kind != .reset }) {
            ConfigRowActions.copied($0, from: entry).map(Clipboard.put)
        }
    }

    /// Why the row has no control. One line, in the machine's own terms.
    static func envOwnedReason(_ entry: ConfigEntry) -> String {
        guard let envVar = entry.envVar else {
            return "Set by this machine's environment, which wins over anything saved here."
        }
        return "Set by \(envVar) on this machine, which wins over anything saved here."
    }

    /// No client-side bound on a typed number -- the 422 is the bound
    /// (design decision 3), and it arrives as the machine's own sentence
    /// under this row.
    func commit(onBlur: Bool) {
        guard let scalar = ConfigValueField.commitScalar(text: text, entry: entry, onBlur: onBlur) else {
            text = entry.editableText
            return
        }
        Task { await onSet(scalar) }
    }
}
