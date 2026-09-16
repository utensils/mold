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

    /// Pins the one format string: what was attempted, on which machine, in
    /// the machine's own words.
    @Test func aFailureNamesTheMachine() {
        let machine = host("plato")
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("no route to host"), on: machine.id,
                     doing: "list its queue")

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.sentence
            == "Couldn't list its queue on plato. Couldn't reach this machine. no route to host")
    }

    /// A drain that retries four times must leave one line, not four.
    @Test func aRetryLeavesOneLinePerMachineAndVerb() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("first attempt"), on: machine.id,
                     doing: "list its queue")
        hosts.report(MoldClientError.unreachable("second attempt"), on: machine.id,
                     doing: "list its queue")

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.sentence.contains("second attempt") == true)
    }

    /// Two different verbs on the same machine are two lines, not one --
    /// unlike a retry of the SAME verb, they say different things.
    @Test func differentVerbsOnOneMachineAreSeparateLines() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its trash")

        #expect(hosts.failures.count == 2)
    }

    @Test func aSuccessClearsWhatThatMachineWasFailingAt() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")
        #expect(!hosts.failures.isEmpty)

        hosts.succeeded(on: machine.id)
        #expect(hosts.failures.isEmpty)
    }

    @Test func dismissingOneFailureLeavesTheOthers() {
        let machine = host()
        let hosts = HostStore(hosts: [machine])
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its queue")
        hosts.report(MoldClientError.unreachable("down"), on: machine.id, doing: "list its trash")

        let toDismiss = hosts.failures[0]
        hosts.dismiss(toDismiss)

        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.id != toDismiss.id)
    }
}
