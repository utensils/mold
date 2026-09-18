import Foundation

/// One fact about a print: what it is called, and what it says.
public struct PrintDetailRow: Equatable, Sendable {
    public let label: String
    public let value: String
    /// Prose rather than a figure. A prompt is a paragraph and wraps; a seed
    /// is a number and sits on the baseline beside its label.
    public let isProse: Bool

    public init(_ label: String, _ value: String, isProse: Bool = false) {
        self.label = label
        self.value = value
        self.isProse = isProse
    }

    /// What its contextual menu calls copying it, the way the Finder names
    /// what it is about.
    public var copyTitle: String { "Copy \(label)" }
}

/// A heading and the facts under it.
public struct PrintDetailGroup: Equatable, Sendable, Identifiable {
    public let title: String
    public let rows: [PrintDetailRow]
    public var id: String { title }
}

/// Everything a finished print records about itself, as rows to read.
///
/// Pure, and away from any view, so what a picture shows, what a clip shows
/// and what a mesh shows is a test rather than something you check by clicking
/// through three prints. A group EXISTS only when it has something in it and a
/// row only when the print recorded that field: an inspector of em-dashes
/// describes a wire format, not a picture.
public enum PrintDetails {
    /// In the order a person reads them: what was asked for, what answered,
    /// how it was steered, what it was made from, and the file at the end.
    public static func groups(for entry: LibraryEntry) -> [PrintDetailGroup] {
        [
            promptGroup(entry.print.metadata),
            modelGroup(entry),
            settingsGroup(entry.print.metadata),
            clipGroup(entry.print),
            meshGroup(entry.print.metadata),
            sourcesGroup(entry.print.metadata),
            sequenceGroup(entry.print.metadata),
            workflowGroup(entry.print.metadata),
            fileGroup(entry),
        ].compactMap(\.self)
    }

    /// A group of the rows that exist, or nothing at all.
    static func group(_ title: String, _ rows: [PrintDetailRow?]) -> PrintDetailGroup? {
        let rows = rows.compactMap(\.self).filter { !$0.value.isEmpty }
        return rows.isEmpty ? nil : PrintDetailGroup(title: title, rows: rows)
    }

    static func row(_ label: String, _ value: String?, isProse: Bool = false) -> PrintDetailRow? {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        return PrintDetailRow(label, value, isProse: isProse)
    }
}
