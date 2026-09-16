import SwiftUI

/// Renaming a tag across the whole library.
struct TagNameSheet: View {
    let tag: String
    let rename: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename Tag").font(.headline)
            Text("Every print carrying \u{201C}\(tag)\u{201D}, on every machine, is renamed.")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField("Name", text: $name)
                .textFieldStyle(.roundedBorder)
                .onSubmit(commit)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Rename", action: commit)
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmed.isEmpty || trimmed == tag)
            }
        }
        .padding(20)
        .frame(width: 380)
        .onAppear { name = tag }
    }

    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    private func commit() {
        guard !trimmed.isEmpty, trimmed != tag else { return }
        rename(trimmed)
        dismiss()
    }
}
