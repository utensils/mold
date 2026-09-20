import MoldClient
import SwiftUI

// The controls themselves, split from `SettingRow.swift` for size: one editor
// per `SettingKey.Editor` case, plus the read-only form an env-owned row draws
// instead of any of them.
extension SettingRow {
    @ViewBuilder var editor: some View {
        if entry.isEnvOwned {
            envOwnedField
        } else {
            switch setting.editor {
            case .toggle: toggleField
            case .text: textField
            case let .number(min, max, step): numberField(min: min, max: max, step: step)
            case let .choice(options): choiceField(options)
            }
        }
    }

    /// The value as plain text, and nothing to drag -- the same shape
    /// `ConfigValueField.envOwnedField` draws in Advanced, so one machine
    /// reads the same either side of the Settings window (review 05-M11).
    var envOwnedField: some View {
        Text(entry.editableText.isEmpty ? "Not set" : entry.editableText)
            .foregroundStyle(.secondary)
    }

    var toggleField: some View {
        Toggle(isOn: toggleBinding) { EmptyView() }.labelsHidden()
    }

    var toggleBinding: Binding<Bool> {
        Binding(
            get: { entry.value == .bool(true) },
            set: { newValue in Task { await onSet(.bool(newValue)) } }
        )
    }

    var textField: some View {
        TextField("", text: $text, prompt: Text("Not set"))
            .labelsHidden()
            .textFieldStyle(.roundedBorder)
            .focused($focused)
            .onSubmit { commit(onBlur: false) }
            .onChange(of: focused) { was, is_ in if was, !is_ { commit(onBlur: true) } }
            .frame(width: 220)
    }

    func numberField(min: Double, max: Double, step: Double?) -> some View {
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

    func stepperBinding(min: Double, max: Double, step: Double) -> Binding<Double> {
        Binding(
            get: { entry.value.double ?? min },
            set: { newValue in Task { await onSet(.number(Swift.min(Swift.max(newValue, min), max))) } }
        )
    }

    func choiceField(_ options: [String]) -> some View {
        Picker("", selection: choiceBinding(options)) {
            ForEach(options, id: \.self) { option in Text(option).tag(option) }
        }
        .labelsHidden()
        .frame(width: 160)
    }

    func choiceBinding(_ options: [String]) -> Binding<String> {
        Binding(
            get: { entry.value.text ?? options.first ?? "" },
            set: { newValue in Task { await onSet(.string(newValue)) } }
        )
    }
}
