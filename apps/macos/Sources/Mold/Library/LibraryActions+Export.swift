import AppKit
import MoldClient
import SwiftUI

// Getting a print out of Mold: what the machine that holds it will convert
// it into, and where the result lands. Split for size.
@MainActor
extension LibraryActions {
    /// What this print can be converted into on the machine that holds it.
    ///
    /// The conversion happens THERE, so the app never needs a decoder for
    /// every container mold can write.
    func exportFormats(for entry: LibraryEntry) -> [String] {
        guard let options = hosts.exportOptions[entry.hostID] else { return [] }
        if entry.print.isMesh { return options.forMesh }
        if entry.print.isVideo { return options.forVideo }
        return []
    }

    /// Converts a print and saves the result.
    func export(_ entry: LibraryEntry, as format: String) {
        Task {
            guard let client = backend(entry.hostID) as? HTTPBackend else { return }
            guard let data = try? await client.export(entry.print.filename, format: format)
            else { return }

            let panel = NSSavePanel()
            let stem = (entry.print.filename as NSString).deletingPathExtension
            panel.nameFieldStringValue = "\(stem).\(format)"
            guard await panel.begin() == .OK, let url = panel.url else { return }
            try? data.write(to: url)
        }
    }
}
