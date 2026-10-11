import Foundation
import MoldClientTesting
import Testing
@testable import MoldClient

struct QueueInputBudgetTests {
    @Test func totalBudgetPreservesDescriptorsAndStopsFetchingBytes() async throws {
        let backend = FakeBackend()
        let inputs = (0..<20).map { QueueInput(index: $0, label: "Image \($0)", preview: true) }
        backend.stub("queueInputs(id:)", returning: inputs)
        backend.stub("queueInputThumbnail(id:index:)", returning: Data(repeating: 1, count: 1024))
        let result = try await backend.queueInputPreviews(id: "j", maximumBytes: 4096)
        #expect(result.map(\.input) == inputs)
        #expect(result.reduce(0) { $0 + ($1.bytes?.count ?? 0) } == 4096)
        #expect(backend.count("queueInputThumbnail(id:index:)") == 4)
        #expect(result.suffix(16).allSatisfy { $0.bytes == nil })
    }
}
