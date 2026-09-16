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

    public init(horizontalSpacing: CGFloat = 8, verticalSpacing: CGFloat = 8,
                alignment: VerticalAlignment = .bottom) {
        self.horizontalSpacing = horizontalSpacing
        self.verticalSpacing = verticalSpacing
        self.alignment = alignment
    }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews,
                             cache: inout ()) -> CGSize {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let rows = Self.rows(of: sizes, within: proposal.width ?? .infinity,
                             spacing: horizontalSpacing)
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
        let rows = Self.rows(of: sizes, within: bounds.width, spacing: horizontalSpacing)

        var index = 0
        var y = bounds.minY
        for row in rows {
            let rowHeight = row.reduce(CGFloat.zero) { Swift.max($0, $1.height) }
            var x = bounds.minX
            for size in row {
                // Bottom alignment so every control sits on one baseline
                // whatever its own height.
                let dy = alignment == .bottom ? rowHeight - size.height : 0
                subviews[index].place(at: CGPoint(x: x, y: y + dy),
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
}
