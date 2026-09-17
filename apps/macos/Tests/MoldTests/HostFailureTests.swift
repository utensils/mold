import Foundation
import MoldClient
import Testing

@testable import Mold

/// `HostStore` is the one funnel every store's failures go through, so these
/// pin the funnel itself rather than any one caller of it.
@MainActor
struct HostFailureTests {
    private func host(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// Pins the one format string: the machine as subject, what it couldn't
    /// do, in the machine's own words.
    @Test func aFailureNamesTheMachine() {
        let machine = host("plato")
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Already running."),
                     on: machine.id, doing: "list its queue")

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.sentence == "plato couldn't list its queue — already running.")
    }

    /// The machine leads the sentence, not the verb.
    @Test func theSentenceNamesTheMachineFirst() {
        let machine = host("plato")
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.http(status: 422, code: nil, message: "No such device."),
                     on: machine.id, doing: "change that GPU")

        #expect(hosts.failures.first?.sentence == "plato couldn't change that GPU — no such device.")
    }

    /// A drain that retries four times must leave one line, not four.
    @Test func aRetryLeavesOneLinePerMachineAndVerb() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.http(status: 503, code: nil, message: "First attempt."),
                     on: machine.id, doing: "list its queue")
        hosts.report(MoldClientError.http(status: 503, code: nil, message: "Second attempt."),
                     on: machine.id, doing: "list its queue")

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.sentence.contains("second attempt") == true)
    }

    /// Two different verbs on the same machine are two lines, not one --
    /// unlike a retry of the SAME verb, they say different things. This is a
    /// property of a real refusal; an unreachable machine collapses instead,
    /// see `anUnreachableMachineLeavesOneLineWhateverItWasAsked`.
    @Test func differentVerbsOnOneMachineAreSeparateLines() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                     on: machine.id, doing: "list its queue")
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                     on: machine.id, doing: "list its trash")

        #expect(hosts.failures.count == 2)
    }

    @Test func aSuccessClearsWhatThatMachineWasFailingAt() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                     on: machine.id, doing: "list its queue")
        #expect(!hosts.failures.isEmpty)

        hosts.succeeded(on: machine.id)
        #expect(hosts.failures.isEmpty)
    }

    @Test func dismissingOneFailureLeavesTheOthers() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                     on: machine.id, doing: "list its queue")
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                     on: machine.id, doing: "list its trash")

        let toDismiss = hosts.failures[0]
        hosts.dismiss(toDismiss)

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.id != toDismiss.id)
    }

    /// An unreachable machine fails every verb for the same one reason -- a
    /// line per verb would just be noise, so whatever else it was asked, it
    /// leaves exactly one line.
    @Test func anUnreachableMachineLeavesOneLineWhateverItWasAsked() {
        let machine = host("192.0.2.1")
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("The request timed out."), on: machine.id,
                     doing: "list its GPUs")
        hosts.report(MoldClientError.unreachable("The request timed out."), on: machine.id,
                     doing: "read its memory use")

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.sentence
            == "192.0.2.1 can't be reached — the request timed out.")
    }

    /// Whichever order they arrive in, the reach line always wins over a
    /// verb-specific one -- but a refusal never erases an existing reach
    /// line, because the machine answering at all is a different fact than
    /// whatever it just refused.
    @Test func aRefusalKeepsItsVerb() {
        let machine = host("plato")
        let refusalFirst = HostStore(hosts: [machine])
        refusalFirst.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                             on: machine.id, doing: "list its queue")
        refusalFirst.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")

        #expect(refusalFirst.failures.count == 1)
        #expect(refusalFirst.failures.first?.verb == HostFailure.reachVerb)

        let reachFirst = HostStore(hosts: [machine])
        reachFirst.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")
        reachFirst.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                           on: machine.id, doing: "list its queue")

        #expect(reachFirst.failures.count == 2)
    }

    /// The moment `HostStore+Reachability` finds a machine answering again,
    /// its reach line is gone -- while an unrelated refusal on the same
    /// machine survives, because that has nothing to do with reachability.
    @Test func aMachineThatAnswersAgainClearsItsReachLine() {
        let machine = host("plato")
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")
        hosts.report(MoldClientError.http(status: 409, code: nil, message: "Busy."),
                     on: machine.id, doing: "change that GPU")
        #expect(hosts.failures.count == 2)

        hosts.succeeded(on: machine.id, doing: HostFailure.reachVerb)

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.verb == "change that GPU")
    }

    /// The same clearing, through the real door: a refresh that finds the
    /// machine up.
    @Test func aRefreshThatFindsTheMachineUpClearsItsReachLine() async {
        let machine = host("plato")
        let backend = FakeBackend(host: machine)
        backend.serverStatus = FakeFixtures.serverStatus()
        backend.capabilityBlock = FakeFixtures.capabilities(events: false)
        backend.exportBlock = FakeFixtures.exportOptions()
        let hosts = HostStore(hosts: [machine]) { _ in backend }
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")

        await hosts.refresh(machine)

        #expect(hosts.failures.isEmpty)
    }

    @Test func anUnauthorizedMachineSaysItNeedsAKey() {
        let machine = host("plato")
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unauthorized, on: machine.id, doing: "list its queue")

        #expect(hosts.failures.first?.sentence
            == "plato couldn't list its queue — it needs an API key. Add one in Settings.")
    }
}
