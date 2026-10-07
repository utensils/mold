import CoreGraphics
import SwiftUI
import Testing

@testable import Mold

/// The Generate inspector's alignment rule, MEASURED rather than eyeballed.
///
/// Regression: every group draws its content in a bare `DisclosureGroup`
/// over a leading `VStack` that never says it is the flexible side, so a group
/// whose widest child is narrow shrinks to that child and SwiftUI centres the
/// lot -- "No adapters installed for this model." sat in the middle of the
/// column while every other line led -- and what does lead starts at the
/// CHEVRON's edge rather than the title's (the owner's screenshot,
/// 2026-09-17).
///
/// Everything below is found by DIFFERENCE: the same section is drawn twice,
/// once with the piece under test and once without it, and the columns that
/// gain ink are that piece's. Nothing is asserted about a pixel whose colour
/// this file did not choose.
@MainActor
struct InspectorSectionLayoutTests {
    private static let width: CGFloat = 320
    private static let height: CGFloat = 60

    /// The colored content marker starts at the title inset. Header text and
    /// separators are ignored so their rendering cannot mask misalignment.
    @Test func aSectionsContentStartsAtTheTitlesInset() {
        #expect(contentLeadingX(Self.bar) == Int(InspectorSection<EmptyView, EmptyView>.titleInset))
    }

    /// The bug itself: a line narrower than the column leads like every other
    /// row instead of floating in the middle of it.
    @Test func aNarrowLineLeadsRatherThanFloatingInTheMiddle() {
        let narrow = contentLeadingX(Self.bar)
        let wide = contentLeadingX(Color.red.frame(width: Self.width, height: 6))
        #expect(narrow == wide)
    }

    /// And a section whose ONLY content is narrow is not itself centred: two
    /// sections of very different widths still lead from one edge.
    @Test func aWholeSectionOfNarrowContentStillLeads() {
        let alone = contentLeadingX(Self.bar)
        let besideAWideRow = contentLeadingX(
            VStack(alignment: .leading, spacing: 2) {
                Self.bar
                Color.red.frame(width: Self.width - 40, height: 6)
            })
        #expect(alone == besideAWideRow)
    }

    @Test func collapsedSectionsHideTheirContentWithAndWithoutAccessories() {
        for expanded in [false, true] {
            let section = InspectorSection("Recent", isExpanded: .constant(expanded)) {
                Button("Refresh") {}
            } content: { Self.bar }
            let empty = InspectorSection("Recent", isExpanded: .constant(expanded)) {
                Button("Refresh") {}
            } content: { EmptyView() }
            #expect((gained(by: section, over: empty) != nil) == expanded)
        }
    }

    // MARK: - Measuring

    /// A narrow, fully opaque mark: no antialiased edge to threshold around,
    /// so its leading column is exact.
    private static var bar: some View { Color.red.frame(width: 6, height: 6) }

    private func contentLeadingX(_ content: some View) -> Int? {
        gained(by: section(title: "", content: content),
               over: section(title: "", content: EmptyView()))
    }

    private func section(title: String, content: some View) -> some View {
        InspectorSection(title, isExpanded: .constant(true)) { content }
    }

    /// The leftmost column inked by `drawn` and not by `plain`.
    private func gained(by drawn: some View, over plain: some View) -> Int? {
        let after = inkedColumns(drawn)
        let before = inkedColumns(plain)
        return after.subtracting(before).min()
    }

    private func inkedColumns(_ view: some View) -> Set<Int> {
        let renderer = ImageRenderer(
            content: view
                .frame(width: Self.width, height: Self.height, alignment: .topLeading)
                .background(.black)
        )
        renderer.scale = 1
        guard let image = renderer.cgImage else { return [] }
        let width = image.width
        let height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &pixels, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return [] }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var columns: Set<Int> = []
        for y in 0..<height {
            for x in 0..<width where !columns.contains(x) {
                let offset = (y * width + x) * 4
                if pixels[offset] > 150 && pixels[offset + 1] < 70 && pixels[offset + 2] < 70 {
                    columns.insert(x)
                }
            }
        }
        return columns
    }
}
