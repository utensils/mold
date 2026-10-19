import Foundation

/// Source tiles share the Library's geometry and keep source identities through reflow.
@MainActor public final class JustifiedSourceLayout {
    public struct Item: Equatable, Sendable {
        public let id: String
        public let aspect: Double
        public init(id: String, aspect: Double) { self.id = id; self.aspect = aspect }
    }
    public struct Tile: Identifiable {
        public let id: String
        public let index: Int
        public let width: Double
    }
    public struct Row: Identifiable {
        public let id: String
        public let items: [Tile]
        public let height: Double
    }
    private var items: [Item] = []
    private var width = 0.0
    private var height = 0.0
    private var rows: [Row] = []
    public private(set) var layouts = 0
    public init() {}

    public func resolve(_ items: [Item], width: Double, targetHeight: Double) -> [Row] {
        guard self.items != items || self.width != width || height != targetHeight else { return rows }
        self.items = items; self.width = width; height = targetHeight; layouts += 1
        rows = JustifiedLayout.rows(aspects: items.map(\.aspect), width: width, targetHeight: targetHeight).map { row in
            Row(id: items[row.id].id, items: row.items.map {
                Tile(id: items[$0.index].id, index: $0.index, width: $0.width)
            }, height: row.height)
        }
        return rows
    }
}
