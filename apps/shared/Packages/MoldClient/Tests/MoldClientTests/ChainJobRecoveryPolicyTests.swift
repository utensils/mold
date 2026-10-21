import Testing
@testable import MoldClient

struct ChainJobRecoveryPolicyTests {
    @Test func stoppedJobsKeepResumeButNeverOfferRunningStop() {
        #expect(ChainJobRecoveryPolicy.canResume(state: .interrupted, ephemeral: true))
        #expect(ChainJobRecoveryPolicy.canResume(state: .failed, ephemeral: false))
        #expect(!ChainJobRecoveryPolicy.canResume(state: .failed, ephemeral: true))
        #expect(!ChainJobRecoveryPolicy.canCancel(state: .failed))
        #expect(!ChainJobRecoveryPolicy.canCancel(state: .interrupted))
        #expect(ChainJobRecoveryPolicy.canCancel(state: .paused))
        #expect(ChainJobRecoveryPolicy.canCancel(state: .running))
        #expect(!ChainJobRecoveryPolicy.canCancel(state: .unknown))
    }
}
