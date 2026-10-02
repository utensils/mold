import MoldClient

/// A sweep applies one range to the selection as it stood before the gesture.
/// Shrinking or reversing that range restores the untouched baseline.
struct LibraryDragSelection {
    let ids: [PrintID]
    let anchor: Int
    let baseline: Set<PrintID>
    let selects: Bool

    init?(ids: [PrintID], start: PrintID, selection: Set<PrintID>) {
        guard let anchor = ids.firstIndex(of: start) else { return nil }
        self.ids = ids
        self.anchor = anchor
        baseline = selection
        selects = !selection.contains(start)
    }

    func selection(through id: PrintID) -> Set<PrintID> {
        guard let end = ids.firstIndex(of: id) else { return baseline }
        let range = Set(ids[min(anchor, end)...max(anchor, end)])
        return selects ? baseline.union(range) : baseline.subtracting(range)
    }
}
