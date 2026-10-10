import Testing
@testable import MoldClient

struct QueueTransferEligibilityTests {
    @Test func transferStopsAtAuthoritativeRenderingState() {
        for state: QueueState in [.queued, .paused, .held] {
            #expect(QueueTransferEligibility.allows(state, reservedProtocol: true))
        }
        for state: QueueState in [.running, .unknown, .failed, .complete, .cancelled, .cancelling] {
            #expect(!QueueTransferEligibility.allows(state, reservedProtocol: true))
        }
        #expect(QueueTransferEligibility.allows(.held, reservedProtocol: false))
        #expect(!QueueTransferEligibility.allows(.queued, reservedProtocol: false))
        #expect(!QueueTransferEligibility.allows(.paused, reservedProtocol: false))
    }
}
