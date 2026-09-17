import Foundation
import MoldClient
import Testing

@testable import Mold

/// Following a clip job: the answers that arrive late, the ones that never
/// arrive, and the one that says it is done.
///
/// **Fails today**: nothing in this app follows an upscale.
@MainActor
struct UpscaleStorePollTests {

    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func fake(for host: MoldHost) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(videoUpscale: true)
        fake.exportBlock = FakeFixtures.exportOptions()
        fake.modelRows = [FakeFixtures.upscaler("real-esrgan-x4plus:fp16", downloaded: true)]
        fake.extras.startedFramewiseAnswer = FakeFixtures.framewiseJob(
            "vup-1", state: "running", total: 124)
        fake.extras.framewiseJobs = [FakeFixtures.framewiseJob("vup-1", state: "running", total: 124)]
        return fake
    }

    private func bench(_ backend: FakeBackend, host: MoldHost,
                       interval: Duration = .milliseconds(1)) async -> (UpscaleStore, HostStore) {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        await hosts.refresh(host)
        let models = ModelStore(hosts: hosts)
        await models.refresh()
        let store = UpscaleStore(hosts: hosts, models: models,
                                 library: LibraryStore(hosts: hosts), interval: interval)
        return (store, hosts)
    }

    private func clip(on host: MoldHost) -> LibraryEntry {
        LibraryEntry(host: host, print: FakeFixtures.print("clip.mp4"))
    }

    private func key(_ host: MoldHost) -> UpscaleStore.Key {
        UpscaleStore.Key(host: host.id, filename: "clip.mp4")
    }

    /// The job finishing between two asks: the store takes the settled answer,
    /// stops asking, and re-reads that machine's gallery -- the bigger clip is
    /// a new row in it and nothing else is going to go and look.
    @Test func aJobThatFinishesBetweenTwoPollsRelistsTheGallery() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.prints = []
        let (store, _) = await bench(backend, host: plato)

        await store.start(clip(on: plato))
        await settle { store.jobs[key(plato)]?.state == .running }
        let listingsBefore = backend.callCount("gallery")

        backend.extras.framewiseJobs = [
            FakeFixtures.framewiseJob("vup-1", state: "completed", done: 124, total: 124),
        ]
        await settle { store.jobs[key(plato)]?.state == .completed }
        await settle { backend.callCount("gallery") > listingsBefore }

        // Settled work is never asked about again. At a 1 ms interval this is
        // fifty chances to ask one more time.
        let asks = backend.callCount("framewiseUpscale")
        try? await Task.sleep(for: .milliseconds(50))
        #expect(backend.callCount("framewiseUpscale") == asks)
    }

    /// The machine going away mid-poll: it stops being asked every 750 ms
    /// forever, and the person is told. The repair is `recover()`, not a
    /// retry loop against a machine that is not there.
    @Test func aMachineThatGoesAwayMidPollStopsBeingAskedAndSaysSo() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, hosts) = await bench(backend, host: plato)

        await store.start(clip(on: plato))
        await settle { store.jobs[key(plato)]?.state == .running }
        backend.plantedErrors["framewiseUpscale"] = MoldClientError.unreachable("it is asleep")

        await settle { !hosts.failures.isEmpty }
        let asks = backend.callCount("framewiseUpscale")
        try? await Task.sleep(for: .milliseconds(50))
        #expect(backend.callCount("framewiseUpscale") == asks, "one failure ends the following")
        #expect(hosts.failures.first?.sentence.contains("asleep") == true)

        // And the job is found again the next time the Library opens.
        backend.plantedErrors["framewiseUpscale"] = nil
        await store.recover(on: plato.id)
        await settle { backend.callCount("framewiseUpscale") > asks }
    }

    /// A stale answer about a job that has already been replaced must never
    /// overwrite its successor.
    ///
    /// The whole sequence, as a person would produce it: a job is running and
    /// being asked about, an ask goes out and hangs, the person cancels and
    /// starts again, and the hung answer -- about the FIRST job, at a frame
    /// count from before the cancel -- finally lands.
    @Test func aStaleAnswerNeverOverwritesANewerJob() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, _) = await bench(backend, host: plato)

        await store.start(clip(on: plato))
        await settle { store.jobs[key(plato)]?.id == "vup-1" }

        // Park the next ask. Its answer is about `vup-1`, and it will not
        // land until after the restart below.
        backend.extras.framewiseHeldOpen = true
        await settle { !backend.extras.framewiseWaiters.isEmpty }

        // Cancel, then ask for a new one -- a second job for the same print.
        let cancelled = FakeFixtures.framewiseJob(
            "vup-1", state: "cancelled", done: 60, total: 124)
        let second = FakeFixtures.framewiseJob("vup-2", state: "running", total: 240)
        backend.extras.framewiseJobs = [cancelled, second]
        await store.transition(key(plato), to: .cancel)
        #expect(store.jobs[key(plato)]?.state == .cancelled)

        backend.extras.startedFramewiseAnswer = second
        await store.start(clip(on: plato))
        #expect(store.jobs[key(plato)]?.id == "vup-2")

        // Now the answer about the FIRST job lands.
        backend.releaseFramewise()
        try? await Task.sleep(for: .milliseconds(30))
        #expect(store.jobs[key(plato)]?.id == "vup-2", "the older answer was dropped")
        #expect(store.jobs[key(plato)]?.totalFrames == 240)
    }

    /// Opening the Library finds the clip upscales already running -- one
    /// listing per machine, never one call per print.
    @Test func openingTheLibraryFindsAJobAlreadyRunning() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.extras.framewiseJobs = [
            FakeFixtures.framewiseJob("vup-old", state: "completed", filename: "done.mp4"),
            FakeFixtures.framewiseJob("vup-live", state: "running", done: 7, total: 97),
            FakeFixtures.framewiseJob("vup-upload", state: "running", filename: "x.mp4"),
        ]
        let (store, _) = await bench(backend, host: plato, interval: .seconds(9))

        await store.recover()

        #expect(backend.callCount("framewiseUpscales") == 1)
        #expect(store.jobs[key(plato)]?.id == "vup-live")
        // A settled job is history, not work in flight.
        #expect(store.jobs[UpscaleStore.Key(host: plato.id, filename: "done.mp4")] == nil)
    }

    /// A machine that says nothing about upscaling is never asked -- an older
    /// host 404s that route, and a listing nobody can use is not worth a
    /// request on every visit to the Library.
    @Test func aMachineThatCannotUpscaleIsNotAskedAtAll() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.capabilityBlock = FakeFixtures.capabilities(videoUpscale: false)
        let (store, _) = await bench(backend, host: plato)

        await store.recover()

        #expect(backend.callCount("framewiseUpscales") == 0)
    }
}
