import Foundation
import Testing
@testable import MoldClient

struct MediaFileReadTests {
    @Test func checksFileSizeBeforeMappingAndPreservesBytes() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let bytes = Data([0, 1, 2, 3])
        try bytes.write(to: file)
        #expect(try ResponseCeiling.readFile(file, ceiling: 4, what: "mesh") == bytes)
        #expect(throws: ResponseCeiling.Exceeded.self) {
            try ResponseCeiling.readFile(file, ceiling: 3, what: "mesh")
        }
    }
}
