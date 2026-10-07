import SwiftUI

/// The ⧉ beside a value worth taking elsewhere.
struct CopyButton: View {
    let what: String
    let value: String

    var body: some View {
        Button("Copy \(what)", systemImage: "document.on.document") {
            Clipboard.put(value)
        }
        .labelStyle(.iconOnly)
        .help("Copy \(what.lowercased()) to the clipboard")
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
    }
}
