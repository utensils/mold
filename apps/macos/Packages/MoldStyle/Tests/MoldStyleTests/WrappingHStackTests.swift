import Foundation
import Testing
import SwiftUI

@testable import MoldStyle

private func size(_ width: CGFloat) -> CGSize { CGSize(width: width, height: 26) }

@Test func itemsThatFitStayOnOneRow() {
    let rows = WrappingHStack.rows(of: [size(100), size(100)], within: 300, spacing: 10)
    #expect(rows.count == 1)
}

@Test func spacingCountsTowardTheWidth() {
    // 100 + 10 + 100 = 210, which does not fit 205.
    let rows = WrappingHStack.rows(of: [size(100), size(100)], within: 205, spacing: 10)
    #expect(rows.count == 2)
}

@Test func anItemWiderThanTheRowStillGetsItsOwnRow() {
    // It cannot be made to fit, so it must not silently join another item and
    // push it off the edge.
    let rows = WrappingHStack.rows(of: [size(50), size(500), size(50)],
                                   within: 200, spacing: 10)
    #expect(rows.count == 3)
    #expect(rows[1] == [size(500)])
}

@Test func everyItemIsPlacedExactlyOnce() {
    let widths: [CGFloat] = [60, 120, 90, 200, 45, 300, 70]
    let rows = WrappingHStack.rows(of: widths.map(size), within: 400, spacing: 12)
    #expect(rows.flatMap(\.self).count == widths.count)
    #expect(rows.flatMap(\.self).map(\.width) == widths)
}

@Test func noRowExceedsTheAvailableWidthUnlessOneItemCannotFit() {
    let widths: [CGFloat] = [60, 120, 90, 45, 70, 110]
    let rows = WrappingHStack.rows(of: widths.map(size), within: 300, spacing: 10)
    for row in rows where row.count > 1 {
        let used = row.map(\.width).reduce(0, +) + 10 * CGFloat(row.count - 1)
        #expect(used <= 300)
    }
}

@Test func noEmptyRowsAreProduced() {
    #expect(WrappingHStack.rows(of: [], within: 300, spacing: 10).isEmpty)
    let rows = WrappingHStack.rows(of: [size(100)], within: 300, spacing: 10)
    #expect(rows.allSatisfy { !$0.isEmpty })
}

@MainActor
@Test func oversizedChipUsesTheProposedWidth() throws {
    let renderer = ImageRenderer(content:
        WrappingHStack {
            HStack {
                Text(String(repeating: "long-tag-", count: 30)).lineLimit(1)
                Image(systemName: "xmark")
            }
            .padding(7)
            Text("Next tag")
        }
    )
    renderer.proposedSize = ProposedViewSize(width: 180, height: nil)
    let image = try #require(renderer.cgImage)
    #expect(image.width <= 180)
    #expect(image.height > 30, "The next tag should wrap onto its own row")
}
