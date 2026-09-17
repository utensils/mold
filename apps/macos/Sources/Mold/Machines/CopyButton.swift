import SwiftUI

/// The ⧉ beside a value worth taking elsewhere.
struct CopyButton: View {
    let what: String
    let value: String

    var body: some View {
        Button("Copy \(what)", systemImage: "document.on.document") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(value, forType: .string)
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
    }
}
