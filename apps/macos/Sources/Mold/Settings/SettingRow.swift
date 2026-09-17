import MoldClient
import SwiftUI

/// One `SettingKey` drawn against a machine's row: `LabeledContent` with the
/// label, the editor, the help as a `.caption` footnote, and the refusal
/// underneath in the machine's own words. `.choice` -> `Picker`, `.number`
/// -> a `TextField` plus a `Stepper` where the key declares a step,
/// `.toggle` -> `Toggle`, `.text` -> `TextField`. Text and number commits
/// share `ConfigValueField`'s blur rule (`commitScalar`) rather than a
/// second copy of it -- a curated key's own `.editor` differs from
/// `ConfigEntry`'s inferred one only for `.choice`, and a choice commits
/// immediately on selection, the same way `ConfigValueField`'s `.toggle`
/// already does.
struct SettingRow: View {
    let setting: SettingKey
    let entry: ConfigEntry
    let refusal: String?
    let onSet: (ConfigScalar) async -> Void

    @State private var text = ""
    @FocusState private var focused: Bool

    /// What one row draws, or nothing -- pure, the same `DiscoverRow.resolve`
    /// idiom `ConfigValueField.Plan` already uses. `entry == nil` is a key
    /// this build curates that the machine's own listing never mentioned --
    /// an older host, or `umt5_variant`'s answer if it were ever curated.
    struct Plan: Equatable {
        let setting: SettingKey
        let entry: ConfigEntry
        let refusal: String?

        /// Whether the number editor draws a `Stepper` beside its field --
        /// only where the key itself declares a step.
        var showsStepper: Bool {
            if case let .number(_, _, step) = setting.editor { return step != nil }
            return false
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
            if let refusal {
                Text(refusal)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .task(id: entry) { text = entry.editableText }
    }

    @ViewBuilder private var editor: some View {
        switch setting.editor {
        case .toggle: toggleField
        case .text: textField
        case let .number(min, max, step): numberField(min: min, max: max, step: step)
        case let .choice(options): choiceField(options)
        }
    }

    private var toggleField: some View {
        Toggle(isOn: toggleBinding) { EmptyView() }.labelsHidden()
    }

    private var toggleBinding: Binding<Bool> {
        Binding(
            get: { entry.value == .bool(true) },
            set: { newValue in Task { await onSet(.bool(newValue)) } }
        )
    }

    private var textField: some View {
        TextField("", text: $text)
            .focused($focused)
            .onSubmit { commit(onBlur: false) }
            .onChange(of: focused) { was, is_ in if was, !is_ { commit(onBlur: true) } }
            .frame(width: 220)
    }

    private func numberField(min: Double, max: Double, step: Double?) -> some View {
        HStack(spacing: 4) {
            TextField("", text: $text)
                .focused($focused)
                .onSubmit { commit(onBlur: false) }
                .onChange(of: focused) { was, is_ in if was, !is_ { commit(onBlur: true) } }
                .frame(width: 80)
            if let step {
                Stepper("", value: stepperBinding(min: min, max: max, step: step), in: min...max, step: step)
                    .labelsHidden()
            }
        }
    }

    private func stepperBinding(min: Double, max: Double, step: Double) -> Binding<Double> {
        Binding(
            get: { entry.value.double ?? min },
            set: { newValue in Task { await onSet(.number(Swift.min(Swift.max(newValue, min), max))) } }
        )
    }

    private func choiceField(_ options: [String]) -> some View {
        Picker("", selection: choiceBinding(options)) {
            ForEach(options, id: \.self) { option in Text(option).tag(option) }
        }
        .labelsHidden()
        .frame(width: 160)
    }

    private func choiceBinding(_ options: [String]) -> Binding<String> {
        Binding(
            get: { entry.value.text ?? options.first ?? "" },
            set: { newValue in Task { await onSet(.string(newValue)) } }
        )
    }

    /// No client-side bound on a typed number -- the 422 is the bound
    /// (design decision 3), and it arrives as the machine's own sentence
    /// under this row.
    private func commit(onBlur: Bool) {
        guard let scalar = ConfigValueField.commitScalar(text: text, entry: entry, onBlur: onBlur) else {
            text = entry.editableText
            return
        }
        Task { await onSet(scalar) }
    }
}
