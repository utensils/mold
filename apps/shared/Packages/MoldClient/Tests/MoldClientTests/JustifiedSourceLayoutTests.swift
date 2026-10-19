import Testing
@testable import MoldClient

@MainActor struct JustifiedSourceLayoutTests {
    @Test func sourceRowsUseLibraryGeometryWithoutSquareLetterboxing() {
        let layout = JustifiedSourceLayout()
        let items = [JustifiedSourceLayout.Item(id: "portrait", aspect: 0.5), .init(id: "wide", aspect: 2), .init(id: "square", aspect: 1)]
        let rows = layout.resolve(items, width: 600, targetHeight: 200)
        let expected = JustifiedLayout.rows(aspects: items.map(\.aspect), width: 600, targetHeight: 200)
        #expect(rows.count == expected.count)
        for (row, reference) in zip(rows, expected) {
            #expect(row.height == reference.height)
            #expect(row.items.map(\.width) == reference.items.map(\.width))
            for tile in row.items {
                #expect(abs(tile.width / row.height - items[tile.index].aspect) < 0.000001)
                #expect(tile.id == items[tile.index].id)
            }
            #expect(row.items.reduce(0) { $0 + $1.width } + Double(row.items.count - 1) * JustifiedLayout.gap <= 600.000001)
        }
        #expect(rows[0].items[0].width != rows[0].height)
    }
    @Test func resizeReflowsOnceAndStableRedrawsKeepGeometry() {
        let layout = JustifiedSourceLayout(), items = (0..<100).map { JustifiedSourceLayout.Item(id: "\($0)", aspect: $0.isMultiple(of: 2) ? 0.6 : 1.8) }
        for _ in 0..<100 { _ = layout.resolve(items, width: 900, targetHeight: 150) }
        #expect(layout.layouts == 1)
        let small = layout.resolve(items, width: 900, targetHeight: 150)
        let large = layout.resolve(items, width: 900, targetHeight: 260)
        #expect(layout.layouts == 2)
        #expect(small.count != large.count)
        #expect(Set(large.flatMap(\.items).map(\.id)) == Set(items.map(\.id)))
    }
}
