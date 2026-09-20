import MoldClient
import SwiftUI

// The Advanced table's four editors, split from `ConfigValueField.swift` for
// size. One per `Kind`, plus the read-only form an env-owned row draws.
extension ConfigValueField {
    var envOwnedField: some View {
        HStack(spacing: 4) {
            Text(entry.editableText.isEmpty ? "Not set" : entry.editableText)
                .foregroundStyle(.secondary)
            if let envVar = plan.envVar {
                Text(envVar).font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    var toggleField: some View {
        Toggle(isOn: toggleBinding) { EmptyView() }
            .labelsHidden()
    }

    var toggleBinding: Binding<Bool> {
        Binding(
            get: { entry.value == .bool(true) },
            set: { newValue in Task { await onSet(.bool(newValue)) } }
        )
    }

    var secretField: some View {
        HStack(spacing: 6) {
            SecureField("", text: $text)
                .accessibilityLabel(entry.key)
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

    var textLikeField: some View {
        TextField(plan.kind == .unset ? "Not set" : "", text: $text)
            .focused($focused)
            .onSubmit { commit(onBlur: false) }
            .onChange(of: focused) { wasFocused, isFocused in
                if wasFocused, !isFocused { commit(onBlur: true) }
            }
    }
}
