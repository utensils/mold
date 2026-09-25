import Foundation
import MoldClient
import Testing

@testable import Mold

/// Pausing a machine's WHOLE queue.
///
/// **Fails today**: `canPauseQueue` is read and no verb is ever called, so
/// `queuePaused` is written only by a frame somebody else's client caused.
@MainActor
struct QueueGateTests {

    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func fake(for host: MoldHost, canPause: Bool = true,
                      paused: Bool? = nil) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus(queuePaused: paused)
        fake.capabilityBlock = FakeFixtures.capabilities(canPauseQueue: canPause)
        fake.exportBlock = FakeFixtures.exportOptions()
        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        return fake
    }

    private func bench(_ backend: FakeBackend, host: MoldHost)
        async -> (QueueGateControl, QueueStore, HostStore) {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        await hosts.refresh(host)
        let queue = QueueStore(hosts: hosts, coalesceDelay: .milliseconds(1))
        return (QueueGateControl(hosts: hosts, queue: queue), queue, hosts)
    }

    /// The gate is READ before it is decided. Desktop's own bug: the first
    /// press paused for real, the value stayed false because nobody had
    /// fetched it, and the second press paused again.
    @Test func theGateIsReadFromTheMachineBeforeItIsDecided() async {
        let workstation = machine()
        let (gate, _, _) = await bench(fake(for: workstation, paused: true), host: workstation)
        #expect(gate.isPaused(on: workstation.id), "the status poll already carried the answer")
    }

    @Test func togglingAPausedMachineResumesIt() async {
        let workstation = machine()
        let backend = fake(for: workstation, paused: true)
        let (gate, _, _) = await bench(backend, host: workstation)

        await gate.toggle(on: workstation.id)

        #expect(backend.extras.gateCalls == [false])
        #expect(backend.callCount("resumeQueue") == 1)
        #expect(!gate.isPaused(on: workstation.id))
    }

    /// The VERB is the writer. What is written is the MACHINE's answer, never
    /// the intent -- a machine that refuses to move leaves the state where it
    /// really is rather than where this app asked for.
    @Test func whatIsWrittenIsTheMachinesAnswerNotTheIntent() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.extras.gateAnswer = false
        let (gate, _, _) = await bench(backend, host: workstation)

        await gate.set(true, on: workstation.id)

        #expect(backend.extras.gateCalls == [true], "it asked to pause")
        #expect(!gate.isPaused(on: workstation.id), "and the machine said it did not")
    }

    /// The frame is the CONFIRMATION. It carries a value rather than a
    /// toggle, so arriving after the verb has already written lands on the
    /// same state -- nothing double-applies.
    @Test func theFrameConfirmsRatherThanTogglingAgain() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let (gate, store, hosts) = await bench(backend, host: workstation)
        await hosts.refresh(workstation)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }

        await gate.set(true, on: workstation.id)
        #expect(gate.isPaused(on: workstation.id))

        backend.emit(.queue(.paused))
        await settle { store.queuePaused[workstation.id] == true }
        #expect(gate.isPaused(on: workstation.id), "still paused, not toggled back")

        // And a frame nobody's verb caused still moves it -- another client
        // pausing this machine is a real thing to hear about.
        backend.emit(.queue(.resumed))
        await settle { store.queuePaused[workstation.id] == false }
        #expect(!gate.isPaused(on: workstation.id))
    }

    /// A resync says deltas were DROPPED -- a `queue_resumed` may be among
    /// them. A cached `true` is preferred over the status poll for as long as
    /// it stands, so without clearing it the pane keeps saying a running
    /// machine is paused.
    ///
    /// **Fails today**: the cached value outlives the gap.
    @Test func aDroppedFrameHandsTheQuestionBackToTheMachine() async {
        let workstation = machine()
        let backend = fake(for: workstation, paused: false)
        let (gate, store, hosts) = await bench(backend, host: workstation)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }

        backend.emit(.queue(.paused))
        await settle { store.queuePaused[workstation.id] == true }
        #expect(gate.isPaused(on: workstation.id))

        backend.emit(.resyncRequired)
        await settle { store.queuePaused[workstation.id] == nil }
        #expect(!gate.isPaused(on: workstation.id), "the machine's own status answers again")
    }

    /// Absent means an older machine with no gate. The control is then ABSENT
    /// -- nothing is sent, and nothing is offered.
    @Test func aMachineThatDoesNotAdvertiseItIsNeverAsked() async {
        let workstation = machine()
        let backend = fake(for: workstation, canPause: false)
        let (gate, store, hosts) = await bench(backend, host: workstation)

        await gate.set(true, on: workstation.id)

        #expect(backend.callCount("pauseQueue") == 0)
        #expect(QueueGateControl.targets(hosts.hosts, capabilities: hosts.capabilities).isEmpty)
    }

    /// A failure is a sentence about the machine, not a silently unchanged
    /// toggle.
    @Test func aRefusalIsSaidOutLoudAndChangesNothing() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.plantedErrors["pauseQueue"] = MoldClientError.http(
            status: 409, code: nil, message: "Another client holds the gate.")
        let (gate, store, hosts) = await bench(backend, host: workstation)

        await gate.set(true, on: workstation.id)

        #expect(!gate.isPaused(on: workstation.id))
        // `HostStore.report` makes the machine the sentence's subject, so the
        // refusal follows it in lower case.
        #expect(hosts.failures.first?.sentence.contains("another client holds the gate") == true)
    }

    // MARK: - What is offered

    private func offer(_ machines: [QueueGateOffer.Machine]) -> QueueGateOffer {
        QueueGateOffer(machines: machines, toggle: { _ in })
    }

    /// The word on the control says what pressing it DOES.
    @Test func oneMachineIsOnePlainItem() {
        let running = offer([.init(id: UUID(), name: "workstation", isPaused: false)])
        #expect(running.items().map(\.title) == ["Pause Queue"])
        let paused = offer([.init(id: UUID(), name: "workstation", isPaused: true)])
        #expect(paused.items().map(\.title) == ["Resume Queue"])
    }

    /// A mixed fleet names each machine, the Empty Queue… idiom -- an
    /// unlabelled item would not say which machine it stops.
    @Test func severalMachinesEachGetNamed() {
        let offered = offer([
            .init(id: UUID(), name: "workstation", isPaused: false),
            .init(id: UUID(), name: "hal9000", isPaused: true),
        ]).items().filter { !$0.isSeparator }.map(\.title)
        #expect(offered == [
            "Pause Queue on All Machines", "Resume Queue on All Machines",
            "Pause Queue on workstation", "Resume Queue on hal9000",
        ])
    }

    /// The fleet verb offered is only the one that would change something.
    @Test func allMachinesOffersOnlyTheVerbThatMovesSomething() {
        let running = offer([
            .init(id: UUID(), name: "workstation", isPaused: false),
            .init(id: UUID(), name: "hal9000", isPaused: false),
        ]).items()
        #expect(running.first?.kind == .all(paused: true))
        #expect(!running.contains { $0.title == QueueGateOffer.resumeAllTitle })
    }

    @Test func pausingEveryMachineSkipsOneAlreadyPaused() async {
        let workstation = machine()
        let hal = machine("hal9000")
        let running = fake(for: workstation)
        let paused = fake(for: hal, paused: true)
        let hosts = HostStore(hosts: [workstation, hal]) { $0.id == workstation.id ? running : paused }
        await hosts.refresh(workstation)
        await hosts.refresh(hal)
        let queue = QueueStore(hosts: hosts, coalesceDelay: .milliseconds(1))
        let gate = QueueGateControl(hosts: hosts, queue: queue)

        await gate.perform(.all(paused: true))

        #expect(running.extras.gateCalls == [true])
        #expect(paused.extras.gateCalls.isEmpty, "already paused -- nothing sent")
        #expect(gate.isPaused(on: workstation.id))
    }

    @Test func noMachineDrawsNothing() {
        #expect(offer([]).items().isEmpty)
    }

    /// The pane says it in a sentence. A toggled label alone is not visible:
    /// it tells you what pressing it does, not what is true right now.
    @Test func aPausedMachineGetsASentenceNotJustALabel() {
        let sentence = QueueGateOffer.pausedSentence(machine: "workstation")
        #expect(sentence.contains("workstation"))
        #expect(sentence.contains("paused"))
        #expect(sentence.contains("not starting anything new"))
    }

    /// Nothing here binds a key. The Library owns a bare Space for Quick
    /// Look, and this app does not bind one chord twice.
    @Test func theGateCarriesNoKeyboardChord() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent()
                .appending(path: "Sources/Mold/Queue/QueueGate.swift"),
            encoding: .utf8)
        #expect(source.count > 500, "the source file was not found")
        #expect(!source.contains("keyboardShortcut"))
        #expect(!source.contains(".space"))
    }
}
