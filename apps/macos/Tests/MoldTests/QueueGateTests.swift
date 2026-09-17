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

    private func machine(_ name: String = "plato") -> MoldHost {
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

    private func bench(_ backend: FakeBackend, host: MoldHost) async -> (QueueStore, HostStore) {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        await hosts.refresh(host)
        return (QueueStore(hosts: hosts, coalesceDelay: .milliseconds(1)), hosts)
    }

    /// The gate is READ before it is decided. Desktop's own bug: the first
    /// press paused for real, the value stayed false because nobody had
    /// fetched it, and the second press paused again.
    @Test func theGateIsReadFromTheMachineBeforeItIsDecided() async {
        let plato = machine()
        let (store, _) = await bench(fake(for: plato, paused: true), host: plato)
        #expect(store.isQueuePaused(on: plato.id), "the status poll already carried the answer")
    }

    @Test func togglingAPausedMachineResumesIt() async {
        let plato = machine()
        let backend = fake(for: plato, paused: true)
        let (store, _) = await bench(backend, host: plato)

        await store.toggleQueuePaused(on: plato.id)

        #expect(backend.extras.gateCalls == [false])
        #expect(backend.callCount("resumeQueue") == 1)
        #expect(!store.isQueuePaused(on: plato.id))
    }

    /// The VERB is the writer. What is written is the MACHINE's answer, never
    /// the intent -- a machine that refuses to move leaves the state where it
    /// really is rather than where this app asked for.
    @Test func whatIsWrittenIsTheMachinesAnswerNotTheIntent() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.extras.gateAnswer = false
        let (store, _) = await bench(backend, host: plato)

        await store.setQueuePaused(true, on: plato.id)

        #expect(backend.extras.gateCalls == [true], "it asked to pause")
        #expect(!store.isQueuePaused(on: plato.id), "and the machine said it did not")
    }

    /// The frame is the CONFIRMATION. It carries a value rather than a
    /// toggle, so arriving after the verb has already written lands on the
    /// same state -- nothing double-applies.
    @Test func theFrameConfirmsRatherThanTogglingAgain() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, hosts) = await bench(backend, host: plato)
        await hosts.refresh(plato)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }

        await store.setQueuePaused(true, on: plato.id)
        #expect(store.isQueuePaused(on: plato.id))

        backend.emit(.queue(.paused))
        await settle { store.queuePaused[plato.id] == true }
        #expect(store.isQueuePaused(on: plato.id), "still paused, not toggled back")

        // And a frame nobody's verb caused still moves it -- another client
        // pausing this machine is a real thing to hear about.
        backend.emit(.queue(.resumed))
        await settle { store.queuePaused[plato.id] == false }
        #expect(!store.isQueuePaused(on: plato.id))
    }

    /// Absent means an older machine with no gate. The control is then ABSENT
    /// -- nothing is sent, and nothing is offered.
    @Test func aMachineThatDoesNotAdvertiseItIsNeverAsked() async {
        let plato = machine()
        let backend = fake(for: plato, canPause: false)
        let (store, hosts) = await bench(backend, host: plato)

        await store.setQueuePaused(true, on: plato.id)

        #expect(backend.callCount("pauseQueue") == 0)
        #expect(QueueStore.gateTargets(hosts.hosts, capabilities: hosts.capabilities).isEmpty)
    }

    /// A failure is a sentence about the machine, not a silently unchanged
    /// toggle.
    @Test func aRefusalIsSaidOutLoudAndChangesNothing() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.plantedErrors["pauseQueue"] = MoldClientError.http(
            status: 409, code: nil, message: "Another client holds the gate.")
        let (store, hosts) = await bench(backend, host: plato)

        await store.setQueuePaused(true, on: plato.id)

        #expect(!store.isQueuePaused(on: plato.id))
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
        let running = offer([.init(id: UUID(), name: "plato", isPaused: false)])
        #expect(running.items().map(\.title) == ["Pause Queue"])
        let paused = offer([.init(id: UUID(), name: "plato", isPaused: true)])
        #expect(paused.items().map(\.title) == ["Resume Queue"])
    }

    /// A mixed fleet names each machine, the Empty Queue… idiom -- an
    /// unlabelled item would not say which machine it stops.
    @Test func severalMachinesEachGetNamed() {
        let offered = offer([
            .init(id: UUID(), name: "plato", isPaused: false),
            .init(id: UUID(), name: "hal9000", isPaused: true),
        ]).items().map(\.title)
        #expect(offered == ["Pause Queue on plato", "Resume Queue on hal9000"])
    }

    @Test func noMachineDrawsNothing() {
        #expect(offer([]).items().isEmpty)
    }

    /// The pane says it in a sentence. A toggled label alone is not visible:
    /// it tells you what pressing it does, not what is true right now.
    @Test func aPausedMachineGetsASentenceNotJustALabel() {
        let sentence = QueueGateOffer.pausedSentence(machine: "plato")
        #expect(sentence.contains("plato"))
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
