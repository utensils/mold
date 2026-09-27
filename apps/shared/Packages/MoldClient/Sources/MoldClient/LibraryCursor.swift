import Foundation

/// How a click or an arrow key changes what is selected.
///
/// A pure function over the laid-out sections, so "down from the last row of a
/// day lands in the next day at the same column" is a unit test rather than
/// something you discover by scrolling.
public struct LibraryCursor: Sendable {
    /// Flattened in the order the grid draws, which is what arrow keys follow.
    public let order: [PrintID]
    /// Where each day starts in `order`, so a row can be found within its day.
    private let sectionStarts: [Int]
    public let columns: Int

    public init(sections: [LibrarySection], columns: Int) {
        var order: [PrintID] = []
        var starts: [Int] = []
        for section in sections {
            starts.append(order.count)
            order.append(contentsOf: section.items.map(\.id))
        }
        self.order = order
        self.sectionStarts = starts
        self.columns = Swift.max(columns, 1)
    }

    public enum Move: Sendable { case left, right, up, down, first, last }

    public enum Modifier: Sendable {
        /// A plain click or arrow: replaces the selection.
        case none
        /// Command: adds or removes one.
        case toggle
        /// Shift: extends from the anchor.
        case extend
    }

    public struct Selection: Hashable, Sendable {
        public var items: Set<PrintID>
        /// Where a shift-extend measures from.
        public var anchor: PrintID?
        /// The one with focus, which is what an arrow key moves.
        public var lead: PrintID?

        public init(items: Set<PrintID> = [], anchor: PrintID? = nil, lead: PrintID? = nil) {
            self.items = items
            self.anchor = anchor
            self.lead = lead
        }

        public static let empty = Selection()

        /// The same selection after the rows were rebuilt, with every id run
        /// through `resolve` -- how a selection follows a print whose tile is
        /// now led by another copy (a Save Locally on it, or a sync landing),
        /// rather than silently pointing at a tile that no longer exists.
        /// `resolve` answering `nil` keeps the id as it was.
        public func remapped(through resolve: (PrintID) -> PrintID?) -> Selection {
            func map(_ id: PrintID) -> PrintID { resolve(id) ?? id }
            return Selection(items: Set(items.map(map)), anchor: anchor.map(map), lead: lead.map(map))
        }
    }

    // MARK: - Clicking

    public func clicking(_ id: PrintID, _ modifier: Modifier,
                         from current: Selection) -> Selection {
        switch modifier {
        case .none:
            Selection(items: [id], anchor: id, lead: id)
        case .toggle:
            toggling(id, from: current)
        case .extend:
            extending(to: id, from: current)
        }
    }

    private func toggling(_ id: PrintID, from current: Selection) -> Selection {
        var items = current.items
        if items.contains(id) {
            items.remove(id)
            // Deselecting the lead must not leave a lead that isn't selected.
            let lead = current.lead == id ? items.first : current.lead
            return Selection(items: items, anchor: id, lead: lead)
        }
        items.insert(id)
        return Selection(items: items, anchor: id, lead: id)
    }

    private func extending(to id: PrintID, from current: Selection) -> Selection {
        guard let anchor = current.anchor,
              let from = order.firstIndex(of: anchor),
              let to = order.firstIndex(of: id)
        else { return Selection(items: [id], anchor: id, lead: id) }
        let range = from <= to ? from...to : to...from
        return Selection(items: Set(order[range]), anchor: anchor, lead: id)
    }

    // MARK: - Arrow keys

    public func moving(_ move: Move, _ modifier: Modifier,
                       from current: Selection) -> Selection {
        guard !order.isEmpty else { return current }
        guard let lead = current.lead, let index = order.firstIndex(of: lead) else {
            let first = order[0]
            return Selection(items: [first], anchor: first, lead: first)
        }
        let target = order[destination(from: index, move: move)]
        // Shift keeps the anchor and grows the run; a bare arrow moves.
        return modifier == .extend
            ? extending(to: target, from: current)
            : Selection(items: [target], anchor: target, lead: target)
    }

    private func destination(from index: Int, move: Move) -> Int {
        let last = order.count - 1
        switch move {
        case .left: return Swift.max(index - 1, 0)
        case .right: return Swift.min(index + 1, last)
        case .first: return 0
        case .last: return last
        case .up, .down:
            // Rows are counted WITHIN a day, because each day starts a new row
            // in the grid -- stepping by `columns` across the whole list would
            // land in the wrong column whenever a day doesn't fill its row.
            let start = sectionStart(containing: index)
            let end = sectionEnd(after: start)
            let offset = index - start
            if move == .up {
                let above = offset - columns
                if above >= 0 { return start + above }
                guard start > 0 else { return index }
                // Into the previous day, keeping the column.
                let previousStart = sectionStart(containing: start - 1)
                let previousCount = start - previousStart
                let column = offset % columns
                let lastRow = (previousCount - 1) / columns
                return previousStart + Swift.min(lastRow * columns + column, previousCount - 1)
            } else {
                let below = offset + columns
                if below < end - start { return start + below }
                guard end <= last else { return index }
                let column = offset % columns
                let nextCount = sectionEnd(after: end) - end
                return end + Swift.min(column, nextCount - 1)
            }
        }
    }

    private func sectionStart(containing index: Int) -> Int {
        sectionStarts.last { $0 <= index } ?? 0
    }

    private func sectionEnd(after start: Int) -> Int {
        sectionStarts.first { $0 > start } ?? order.count
    }
}
