import MoldClient
import SwiftUI

/// One row's editor, drawn from `ConfigEntry.Editor` -- the Advanced table's
/// whole inference, since nothing on the wire declares a type
/// (`ConfigEntry+Editing.swift`). Commits on Return and on focus loss, never
/// on appearance, and takes `onSet` as a closure so this view is testable
/// with no store at all -- the call site wires it to `ConfigStore.set`.
struct ConfigValueField: View {
    let entry: ConfigEntry
    let onSet: (ConfigScalar) async -> Void

    @State private var text = ""
    @FocusState private var focused: Bool

    /// What one row draws, from the entry alone -- pure, the same
    /// `DiscoverRow.resolve` idiom the Models table's State column uses.
    struct Plan: Equatable {
        let kind: Kind
        /// `entry.envVar`, but only where it is meant to be shown -- an
        /// env-owned row draws the variable name beside its value.
        let envVar: String?
        let showsReset: Bool
    }

    enum Kind: Equatable {
        case toggle, number, text, unset, secret
        /// `source == "env"`: the value as plain text, no field at all --
        /// PUT would answer 403 `ENV_OVERRIDDEN` before it tried.
        case envOwned
    }

    static func resolve(_ entry: ConfigEntry) -> Plan {
        let kind: Kind = entry.isEnvOwned ? .envOwned : Kind(entry.editor)
        return Plan(kind: kind, envVar: entry.envVar, showsReset: entry.canReset)
    }

    private var plan: Plan { Self.resolve(entry) }

    var body: some View {
        Group {
            switch plan.kind {
            case .envOwned: envOwnedField
            case .toggle: toggleField
            case .secret: secretField
            case .number, .text, .unset: textLikeField
            }
        }
        .task(id: entry) { text = entry.editableText }
        .accessibilityLabel(entry.key)
    }

    private var envOwnedField: some View {
        HStack(spacing: 4) {
            Text(entry.editableText.isEmpty ? "Not set" : entry.editableText)
                .foregroundStyle(.secondary)
            if let envVar = plan.envVar {
                Text(envVar).font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private var toggleField: some View {
        Toggle(isOn: toggleBinding) { EmptyView() }
            .labelsHidden()
    }

    private var toggleBinding: Binding<Bool> {
        Binding(
            get: { entry.value == .bool(true) },
            set: { newValue in Task { await onSet(.bool(newValue)) } }
        )
    }

    private var secretField: some View {
        HStack(spacing: 6) {
            SecureField(entry.secretState, text: $text)
                .focused($focused)
                .onSubmit { commit(onBlur: false) }
            Text(entry.secretState).font(.caption).foregroundStyle(.secondary)
            // The explicit clear an empty Return is no longer allowed to be
            // (review 05-M12), and the same shape `Shell/AccountsRow.swift`
            // already offers for a catalog token: present only where there is
            // something stored to remove.
            if entry.value != .null {
                Button("Clear") { Task { await onSet(.null) } }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
            }
        }
        .onChange(of: focused) { wasFocused, isFocused in
            if wasFocused, !isFocused { commit(onBlur: true) }
        }
    }

    private var textLikeField: some View {
        TextField(plan.kind == .unset ? "Not set" : "", text: $text)
            .focused($focused)
            .onSubmit { commit(onBlur: false) }
            .onChange(of: focused) { wasFocused, isFocused in
                if wasFocused, !isFocused { commit(onBlur: true) }
            }
    }

    /// What one commit sends, or nothing. A non-number reverts rather than
    /// sends -- there is no client-side bound, the 422 is the bound, and it
    /// arrives as the machine's own sentence (design decision 3). A BLUR
    /// sends only what changed: a secret's field starts empty because
    /// "<set>" is a mask, and the parser reads an empty string as `.null`,
    /// so a blur that committed unconditionally would clear the API key on
    /// the machine for anyone who clicked into the field and away again.
    ///
    /// An EMPTY secret is now a no-op on Return too. A secret field always
    /// renders empty, so nothing at all distinguished "I typed nothing" from
    /// "clear it": tab into `runpod.api_key` in the Advanced table, hesitate,
    /// press Return, and the credential was gone (review 05-M12). Clearing is
    /// the row's own Clear button, which says so.
    static func commitScalar(text: String, entry: ConfigEntry, onBlur: Bool) -> ConfigScalar? {
        if onBlur, text == entry.editableText { return nil }
        if entry.editor == .secret, text.isEmpty { return nil }
        return ConfigEntry.scalar(from: text, editor: entry.editor)
    }

    private func commit(onBlur: Bool) {
        guard let scalar = Self.commitScalar(text: text, entry: entry, onBlur: onBlur) else {
            text = entry.editableText
            return
        }
        Task { await onSet(scalar) }
    }
}

private extension ConfigValueField.Kind {
    init(_ editor: ConfigEntry.Editor) {
        switch editor {
        case .toggle: self = .toggle
        case .number: self = .number
        case .text: self = .text
        case .secret: self = .secret
        case .unset: self = .unset
        }
    }
}
