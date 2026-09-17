#if DEBUG
import Testing
@testable import Mold

@MainActor struct UATScriptTests {
    @Test func blankLinesAndCommentsAreNotSteps() {
        let script = "# open the library\n\n  menu View > Library  \nsnapshot /tmp/a.png\n"
        #expect(UATScript.steps(in: script) == ["menu View > Library", "snapshot /tmp/a.png"])
    }

    @Test func withoutTheVariableNothingRuns() {
        UATScript.runIfRequested(environment: [:])
    }
}
#endif
