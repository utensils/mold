import Foundation

/// Ordered, uncropped contact-sheet geometry. Keep in sync with studio/lib/justifiedLayout.ts.
/// Completed rows fill the width; an incomplete final row is left aligned at
/// target height. Missing dimensions are square until metadata changes, never
/// until image decode. Valid extreme ratios are not clamped or cropped.
public enum JustifiedLayout {
    public static let gap: Double = 2

    public struct Tile: Hashable, Sendable {
        public let index: Int
        public let x: Double
        public let width: Double
    }
    public struct Row: Hashable, Sendable, Identifiable {
        public var id: Int { items[0].index }
        public let items: [Tile]
        public let height: Double
        public let top: Double
    }
    public static func aspect(width: Int?, height: Int?) -> Double {
        guard let width, let height, width > 0, height > 0 else { return 1 }
        return Double(width) / Double(height)
    }
    public static func rows(aspects: [Double], width: Double, targetHeight: Double) -> [Row] {
        guard width.isFinite, width > 0, targetHeight.isFinite, targetHeight > 0 else { return [] }
        let ratios = aspects.map { $0.isFinite && $0 > 0 ? $0 : 1 }
        let gap = min(Self.gap, width / 2)
        var rows: [Row] = []
        var start = 0
        var sum = 0.0
        var top = 0.0
        func flush(_ end: Int, justify: Bool) {
            guard end > start else { return }
            let fit = max(0, width - gap * Double(end - start - 1)) / sum
            let height = justify ? fit : min(targetHeight, fit)
            var x = 0.0
            let tiles = (start..<end).map { index in
                let tile = Tile(index: index, x: x, width: ratios[index] * height)
                x += tile.width + gap
                return tile
            }
            rows.append(Row(items: tiles, height: height, top: top))
            top += height + gap
            start = end
            sum = 0
        }
        for index in ratios.indices {
            // Very thin portraits can otherwise fill a row with seams alone.
            if index > start, gap * Double(index - start) >= width { flush(index, justify: false) }
            let fit = (width - gap * Double(index - start)) / (sum + ratios[index])
            if index > start, fit <= targetHeight {
                let previous = (width - gap * Double(index - start - 1)) / sum
                if previous <= targetHeight * 1.5, abs(previous - targetHeight) < abs(fit - targetHeight) {
                    flush(index, justify: true)
                }
            }
            sum += ratios[index]
            if sum * targetHeight + gap * Double(index - start) >= width { flush(index + 1, justify: true) }
        }
        flush(ratios.count, justify: false)
        return rows
    }
}

/// Cache geometry separately from selection, visibility and thumbnail updates.
/// A redraw may compare the section value but must not allocate 30K new tiles.
@MainActor public final class JustifiedLibraryLayout {
    /// Cell identity follows the print when deletion or sorting changes its
    /// section index; thumbnail state must never transfer to its replacement.
    public struct PrintTile: Identifiable {
        public let id: PrintID
        public let index: Int
        public let x: Double
        public let width: Double
    }
    /// SwiftUI's lazy scroll targets take the ForEach identity, not a nested
    /// view's .id modifier. Use the first print so visibility is typed PrintID.
    public struct PrintRow: Identifiable {
        public let id: PrintID
        public let items: [PrintTile]
        public let height: Double
        public let top: Double
    }
    public struct Section: Identifiable {
        public let source: LibrarySection
        public let rows: [JustifiedLibraryLayout.PrintRow]
        public var id: String { source.id }
    }
    private var source: [LibrarySection] = []
    private var width = 0.0
    private var height = 0.0
    private var sections: [Section] = []
    public init() {}
    public func resolve(_ source: [LibrarySection], width: Double, targetHeight: Double) -> [Section] {
        if self.width == width, height == targetHeight, self.source == source { return sections }
        self.source = source
        self.width = width
        height = targetHeight
        sections = source.map { section in
            Section(source: section, rows: JustifiedLayout.rows(aspects: section.items.map {
                JustifiedLayout.aspect(width: $0.print.metadata.width, height: $0.print.metadata.height)
            }, width: width, targetHeight: targetHeight).map { row in
                PrintRow(id: section.items[row.id].id, items: row.items.map {
                    PrintTile(id: section.items[$0.index].id, index: $0.index, x: $0.x, width: $0.width)
                }, height: row.height, top: row.top)
            })
        }
        return sections
    }
}
