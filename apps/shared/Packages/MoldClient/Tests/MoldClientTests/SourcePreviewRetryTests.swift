import Testing
@testable import MoldClient

@MainActor struct SourcePreviewRetryTests {
    @Test func aTransientMissRecoversWithoutRecreatingTheTile() async {
        var attempts = 0
        let result = await SourcePreviewRetry.load(delay: .zero, isCurrent: { true }) {
            attempts += 1
            return attempts == 2 ? "pixels" : nil
        }
        #expect(result == "pixels")
        #expect(attempts == 2)
    }
    @Test func failuresAreBoundedAndInvalidationStopsOldWork() async {
        var attempts = 0
        let result: String? = await SourcePreviewRetry.load(delay: .zero, isCurrent: { true }) {
            attempts += 1; return nil
        }
        #expect(result == nil)
        #expect(attempts == 3)
        attempts = 0
        let stale = await SourcePreviewRetry.load(delay: .zero, isCurrent: { attempts == 0 }) {
            attempts += 1; return "old pixels"
        }
        #expect(stale == nil)
        #expect(attempts == 1)
    }
}
