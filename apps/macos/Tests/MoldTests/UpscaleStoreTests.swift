import Foundation
import MoldClient
import Testing

@testable import Mold

/// Following a clip upscale is the whole of this store, and every way it can
/// go wrong is a SEQUENCE: a second press, a cancel landing before the id
/// does, a job that settles between two asks, a machine that goes away
/// mid-poll, and an answer about a job that has already been replaced.
///
/// **Fails today**: nothing in this app starts or follows an upscale.
@MainActor
struct UpscaleStoreTests {

    // MARK: - The bench

    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// A machine that upscales both kinds and has Real-ESRGAN installed.
    private func fake(for host: MoldHost, stills: Bool = true, clips: Bool = true) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(videoUpscale: clips, galleryImage: stills)
        fake.exportBlock = FakeFixtures.exportOptions()
        fake.modelRows = [FakeFixtures.upscaler("real-esrgan-x4plus:fp16", downloaded: true)]
        fake.extras.startedFramewiseAnswer = running
        fake.extras.framewiseJobs = [running]
        fake.extras.stillUpscaleAnswer = FakeFixtures.stillUpscale("still-4x.png")
        return fake
    }

    private var running: VideoUpscaleJob {
        FakeFixtures.framewiseJob("vup-1", state: "running", total: 124)
    }

    /// A store with its machines already probed and its models listed, which
    /// is what the capability gate and the upscaler picker read.
    private func bench(_ backend: FakeBackend, host: MoldHost, interval: Duration = .milliseconds(1))
        async -> (UpscaleStore, HostStore) {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        await hosts.refresh(host)
        let models = ModelStore(hosts: hosts)
        await models.refresh()
        let library = LibraryStore(hosts: hosts)
        let store = UpscaleStore(hosts: hosts, models: models, library: library,
                                 interval: interval)
        return (store, hosts)
    }

    private func entry(_ filename: String, on host: MoldHost) -> LibraryEntry {
        LibraryEntry(host: host, print: FakeFixtures.print(filename))
    }

    // MARK: - What is offered

    /// Two answers out of one block: a clip needs `video_upscale`, a still
    /// needs `gallery_image` as well (`types.rs:12430-12433`).
    @Test func aStillNeedsTheGalleryRouteAndAClipDoesNot() async {
        let plato = machine()
        let backend = fake(for: plato, stills: false)
        let (store, _) = await bench(backend, host: plato)
        #expect(store.canUpscale(entry("clip.mp4", on: plato)))
        #expect(!store.canUpscale(entry("still.png", on: plato)))
    }

    /// Absence of the whole block is a definitive no -- the action is then
    /// absent from the menu, never present and inert.
    @Test func aMachineThatSaysNothingOffersNothing() async {
        let plato = machine()
        let backend = fake(for: plato, stills: false, clips: false)
        let (store, _) = await bench(backend, host: plato)
        #expect(!store.canUpscale(entry("clip.mp4", on: plato)))
        #expect(!store.canUpscale(entry("still.png", on: plato)))
        #expect(!store.canUpscale(entry("shape.glb", on: plato)))
    }

    /// Nothing reads `/api/models` on the way to the Library, so the model
    /// cache is EMPTY there -- a local "is an upscaler installed" check
    /// refused a machine that had one, in a machine-failure banner, naming
    /// the wrong problem.
    ///
    /// The ported policy already answers for a machine whose upscalers this
    /// app has never listed: its last fallback is the manifest name
    /// (`upscale.ts:24`), and the HOST is the authority on whether it has it.
    ///
    /// **Fails today**: the `isDownloaded` guard makes that fallback
    /// unreachable and reports `NoUpscalerInstalled` instead.
    @Test func aMachineWhoseModelsWereNeverListedIsStillAsked() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        await hosts.refresh(plato)
        // Deliberately NO `models.refresh()` -- this is the Library path.
        let store = UpscaleStore(hosts: hosts, models: ModelStore(hosts: hosts),
                                 library: LibraryStore(hosts: hosts),
                                 interval: .seconds(9))

        await store.start(entry("clip.mp4", on: plato))

        #expect(backend.extras.startedFramewise.map(\.model) == ["real-esrgan-x4plus:fp16"])
        #expect(hosts.failures.isEmpty, "nothing failed, so nothing is said about the machine")
    }

    /// And a machine that really has none refuses in ITS OWN words, which
    /// name the model it could not find -- not this app's guess at why.
    @Test func aRefusalIsTheMachinesOwnSentence() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.plantedErrors["startFramewiseUpscale"] = MoldClientError.http(
            status: 404, code: nil, message: "Unknown upscaler model real-esrgan-x4plus:fp16")
        let (store, hosts) = await bench(backend, host: plato)

        await store.start(entry("clip.mp4", on: plato))

        // `HostStore.report` makes the machine the subject, so the host's
        // own clause follows it in lower case.
        #expect(hosts.failures.first?.sentence.contains("unknown upscaler model") == true)
    }

    // MARK: - The sequences

    /// A still is synchronous: one call, and the machine's gallery is re-read
    /// because the bigger picture is a new row in it.
    @Test func aStillIsOneCallAndARelist() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.prints = []
        let (store, _) = await bench(backend, host: plato)

        await store.start(entry("still.png", on: plato))

        #expect(backend.extras.upscaledStills.map(\.filename) == ["still.png"])
        #expect(backend.extras.upscaledStills.first?.model == "real-esrgan-x4plus:fp16")
        #expect(backend.callCount("gallery") >= 1)
        #expect(store.jobs.isEmpty, "a still leaves no job to follow")
    }

    /// Pressing it twice is ONE request. The second press used to start a
    /// second 124-frame job against the same print, and the first job's id
    /// was lost the moment the second answered.
    @Test func asecondRequestWhileOneIsRunningIsNotASecondJob() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.delays["startFramewiseUpscale"] = .milliseconds(30)
        let (store, _) = await bench(backend, host: plato, interval: .milliseconds(50))
        let clip = entry("clip.mp4", on: plato)

        async let first: Void = store.start(clip)
        await settle { store.isWorking(on: clip) }
        await store.start(clip)
        await first

        #expect(backend.callCount("startFramewiseUpscale") == 1)
        // And still one once the job is being followed rather than started.
        await store.start(clip)
        #expect(backend.callCount("startFramewiseUpscale") == 1)
    }

    /// A cancel during the await. There is no id yet, so the cancel is
    /// remembered and spent the instant the host hands one over -- otherwise
    /// the job upscales every frame on a machine nobody is watching.
    @Test func aCancelBeforeTheIdArrivesStillReachesTheJob() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.delays["startFramewiseUpscale"] = .milliseconds(30)
        let (store, _) = await bench(backend, host: plato, interval: .seconds(9))
        let clip = entry("clip.mp4", on: plato)

        async let started: Void = store.start(clip)
        await settle { store.isWorking(on: clip) }
        await store.transition(UpscaleStore.Key(host: plato.id, filename: "clip.mp4"), to: .cancel)
        await started

        await settle { !backend.extras.framewiseTransitions.isEmpty }
        #expect(backend.extras.framewiseTransitions.map(\.id) == ["vup-1"])
        #expect(backend.extras.framewiseTransitions.first?.to == .cancel)
    }
}
