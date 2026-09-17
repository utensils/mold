import SwiftUI

/// A row that flows onto the next line when it runs out of width.
///
/// The control row grows and shrinks with the recipe -- a video model adds a
/// Length control, an img2img one adds Strength -- so a fixed `HStack` either
/// clips on a narrow window or forces the panel wider than the picture behind
/// it.
public struct WrappingHStack: Layout {
    public let horizontalSpacing: CGFloat
    public let verticalSpacing: CGFloat
    public let alignment: VerticalAlignment
    /// When true, the LAST subview rides the trailing edge of the final row
    /// instead of wrapping like every other item -- the actions row under a
    /// control capsule, which should never sit flush against the last
    /// control (M8 decision 1).
    public let pinsLast: Bool

    public init(horizontalSpacing: CGFloat = 8, verticalSpacing: CGFloat = 8,
                alignment: VerticalAlignment = .bottom, pinsLast: Bool = false) {
        self.horizontalSpacing = horizontalSpacing
        self.verticalSpacing = verticalSpacing
        self.alignment = alignment
        self.pinsLast = pinsLast
    }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews,
                             cache: inout ()) -> CGSize {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let (rows, _) = Self.layout(of: sizes, within: proposal.width ?? .infinity,
                                    spacing: horizontalSpacing, pinsLast: pinsLast)
        var width: CGFloat = 0
        var height: CGFloat = 0
        for row in rows {
            let items: CGFloat = row.reduce(CGFloat.zero) { $0 + $1.width }
            let gaps: CGFloat = horizontalSpacing * CGFloat(max(row.count - 1, 0))
            width = Swift.max(width, items + gaps)
            height += row.reduce(CGFloat.zero) { Swift.max($0, $1.height) }
        }
        height += verticalSpacing * CGFloat(max(rows.count - 1, 0))
        return CGSize(width: width, height: height)
    }

    public func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize,
                              subviews: Subviews, cache: inout ()) {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let (rows, lastIsTrailing) = Self.layout(of: sizes, within: bounds.width,
                                                 spacing: horizontalSpacing, pinsLast: pinsLast)

        var index = 0
        var y = bounds.minY
        for (rowIndex, row) in rows.enumerated() {
            let rowHeight = row.reduce(CGFloat.zero) { Swift.max($0, $1.height) }
            var x = bounds.minX
            let isLastRow = rowIndex == rows.count - 1
            for (itemIndex, size) in row.enumerated() {
                // Bottom alignment so every control sits on one baseline
                // whatever its own height.
                let dy = alignment == .bottom ? rowHeight - size.height : 0
                let isPinned = lastIsTrailing && isLastRow && itemIndex == row.count - 1
                let x0 = isPinned ? bounds.maxX - size.width : x
                subviews[index].place(at: CGPoint(x: x0, y: y + dy),
                                      proposal: ProposedViewSize(size))
                x += size.width + horizontalSpacing
                index += 1
            }
            y += rowHeight + verticalSpacing
        }
    }

    /// Packs sizes into rows. Factored out and `nonisolated` so the wrapping
    /// rule can be unit-tested with no view hierarchy at all.
    public nonisolated static func rows(of sizes: [CGSize], within width: CGFloat,
                                        spacing: CGFloat) -> [[CGSize]] {
        var rows: [[CGSize]] = []
        var row: [CGSize] = []
        var used: CGFloat = 0

        for size in sizes {
            let needed = row.isEmpty ? size.width : used + spacing + size.width
            if !row.isEmpty, needed > width {
                rows.append(row)
                row = [size]
                used = size.width
            } else {
                row.append(size)
                used = needed
            }
        }
        if !row.isEmpty { rows.append(row) }
        return rows
    }

    /// `rows`, with the option to keep the LAST size off the ordinary wrap
    /// and instead ride the trailing edge of the final row -- joining it
    /// when it fits after that row's own items plus one spacing, or taking
    /// its own right-aligned row when it does not.
    public nonisolated static func layout(
        of sizes: [CGSize], within width: CGFloat, spacing: CGFloat, pinsLast: Bool
    ) -> (rows: [[CGSize]], lastIsTrailing: Bool) {
        guard pinsLast, let last = sizes.last else {
            return (rows(of: sizes, within: width, spacing: spacing), false)
        }
        var wrapped = rows(of: Array(sizes.dropLast()), within: width, spacing: spacing)
        guard var lastRow = wrapped.last else {
            return ([[last]], true)
        }
        let used = lastRow.reduce(CGFloat.zero) { $0 + $1.width }
            + spacing * CGFloat(max(lastRow.count - 1, 0))
        if used + spacing + last.width <= width {
            lastRow.append(last)
            wrapped[wrapped.count - 1] = lastRow
        } else {
            wrapped.append([last])
        }
        return (wrapped, true)
    }
}
