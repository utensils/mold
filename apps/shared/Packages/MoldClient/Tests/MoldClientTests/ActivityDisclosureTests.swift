import Testing
@testable import MoldClient

struct ActivityDisclosureTests {
    @Test func progressDoesNotReopenDismissedOperationButNewIssuesAndRunsDo() {
        var state = ActivityDisclosure()
        state.dismiss("sync:one")
        #expect(!state.shows("sync:one"))
        // Progress and ordinary completion share the operation identity.
        let normalCompletionIdentity = "sync:one"
        #expect(!state.shows(normalCompletionIdentity))
        #expect(state.shows("sync:two"))
        #expect(state.shows("sync:one:issue:missing-source"))
        state.restore()
        #expect(state.shows("sync:one"))
        #expect(!state.hasDismissed)
    }
}
