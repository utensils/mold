import Foundation
import Testing
@testable import MoldClient

struct JustifiedLayoutTests {
    @MainActor @Test func scrollTargetsUsePrintIdentityAcrossReflow() throws {
        let host = MoldHost(id: UUID(), name: "Fixture", baseURL: URL(string: "http://localhost")!)
        let meta = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8))
        let entries = (0..<30).map { index in
            LibraryEntry(host: host, print: GalleryPrint(
                filename: "\(index).png", metadata: meta, timestamp: 1, format: "png", sizeBytes: 1,
                mediaVersion: "v", title: nil, tags: nil, favorite: nil, collections: nil,
                trashedAt: nil, purgeAt: nil))
        }
        let cache = JustifiedLibraryLayout()
        for width in [400.0, 800.0] {
            let sections = cache.resolve(LibraryGrouping.byDay(entries), width: width, targetHeight: 180)
            for section in sections {
                for row in section.rows {
                    let identity: PrintID = row.id
                    #expect(identity == section.source.items[row.items[0].index].id)
                }
            }
        }
    }
    @Test func continuousRows() {
        let ratios = [0.5, 1.5, 1, 2, 0.6, 1, 1.6, 0.7]
        let rows = JustifiedLayout.rows(aspects: ratios, width: 600, targetHeight: 180)
        #expect(rows.flatMap { $0.items.map(\.index) } == Array(ratios.indices))
        for (index, row) in rows.enumerated() {
            let width = row.items.reduce(0) { $0 + $1.width } + Double(row.items.count - 1) * 2
            #expect(width <= 600.00001)
            if index < rows.count - 1 { #expect(abs(width - 600) < 0.00001) }
            for tile in row.items { #expect(abs(tile.width / row.height - ratios[tile.index]) < 0.00001) }
        }
    }
    @Test func widowsAndExtremes() {
        #expect(JustifiedLayout.rows(aspects: [0.5], width: 600, targetHeight: 180)[0].height == 180)
        #expect(JustifiedLayout.rows(aspects: [40], width: 600, targetHeight: 180)[0].height == 15)
        #expect(JustifiedLayout.aspect(width: -1, height: 100) == 1)
        #expect(JustifiedLayout.aspect(width: 1, height: 10000) == 0.0001)
        #expect(JustifiedLayout.rows(aspects: [1], width: 0, targetHeight: 180).isEmpty)
    }
    @Test func largeLibrary() {
        let rows = JustifiedLayout.rows(aspects: Array(repeating: 1.5, count: 30000), width: 1000, targetHeight: 180)
        #expect(rows.flatMap(\.items).count == 30000)
        #expect(rows.allSatisfy { $0.height > 0 && $0.height <= 270 })
    }
}
